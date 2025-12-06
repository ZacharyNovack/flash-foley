import argparse
import gc
import json
import os
import csv
import torch
import torchaudio
import copy
import math
import time
from pathlib import Path

from accelerate import Accelerator

from einops import rearrange
from torch.nn import functional as F
from torchaudio import transforms as T

from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.pretrained import get_pretrained_model
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.models.utils import copy_state_dict
from pre_encode import pesto_pitch_256, a_weighted_gpu, spectral_centroid

import scipy
import torch


###############################################################################
# Pitch unit conversions
###############################################################################


def bins_to_cents(bins):
    """Converts pitch bins to cents"""
    cents = 20 * bins + 1997.3794084376191

    # Trade quantization error for noise
    return dither(cents)


def bins_to_frequency(bins):
    """Converts pitch bins to frequency in Hz"""
    return cents_to_frequency(bins_to_cents(bins))


def cents_to_bins(cents, quantize_fn=torch.floor):
    """Converts cents to pitch bins"""
    bins = (cents - 1997.3794084376191) / 20
    return quantize_fn(bins).int()


def cents_to_frequency(cents):
    """Converts cents to frequency in Hz"""
    return 10 * 2 ** (cents / 1200)


def frequency_to_bins(frequency, quantize_fn=torch.floor):
    """Convert frequency in Hz to pitch bins"""
    return cents_to_bins(frequency_to_cents(frequency), quantize_fn)


def frequency_to_cents(frequency):
    """Convert frequency in Hz to cents"""
    return 1200 * torch.log2(frequency / 10.)


###############################################################################
# Utilities
###############################################################################


def dither(cents):
    """Dither the predicted pitch in cents to remove quantization error"""
    noise = scipy.stats.triang.rvs(c=0.5,
                                   loc=-20,
                                   scale=40,
                                   size=cents.size())
    return cents + cents.new_tensor(noise)



def extract_features(audio_tensor, sample_rate=44100):
            
    features = {}
    with torch.no_grad():
        # Pitch
        for control_name, extract_func in {
            "pesto_pitch": pesto_pitch_256,
            "rms_energy": a_weighted_gpu,
            "spectral_centroid": spectral_centroid
        }.items():
            if control_name == "pesto_pitch":
                pitch_preds, control_value = extract_func(audio_tensor, return_predictions=True)
                eval_pitch_d = {
                    'pitch': pitch_preds.cpu(),
                    'activations': control_value.clone().cpu()
                }
                features['eval_pitch'] = eval_pitch_d
            else:
                control_value = extract_func(audio_tensor)
            control_value = control_value.squeeze()  # add batch dim
            if control_value.ndim == 1:
                # add 2 dims
                control_value = control_value.unsqueeze(0).unsqueeze(0)
            elif control_value.ndim == 2:
                # add channel dim
                control_value = control_value.unsqueeze(0)
            # print(f"{control_name} feature shape before interpolation: {control_value.shape}")
            control_value = torch.nan_to_num(control_value, nan=0.0, posinf=0.0, neginf=0.0)
            control_value = F.interpolate(control_value, size=256, mode="linear").squeeze()
            if control_value.ndim == 1:
                control_value = control_value.unsqueeze(0)
            # print(f"{control_name} feature shape after interpolation: {control_value.shape}")
            features[control_name] = control_value

    return features

sample_rate = 32000
sample_size = 1920000

def periodicity(pitch, activations):
    '''
    Periodicity is the activation of the corresponding pitch bin for each frame
    - pitch is sequence of bin indices
    - activations is shape [B x num bins x time]
    '''
    # print(pitch.shape, activations.shape)
    
    periodicity = torch.gather(activations, 1, pitch.unsqueeze(1).long()).squeeze(1)
    return periodicity

def load_model(model_config=None, model_ckpt_path=None, pretrained_name=None, pretransform_ckpt_path=None, model_half=False):
    global model, sample_rate, sample_size
    
    if pretrained_name is not None:
        print(f"Loading pretrained model {pretrained_name}")
        model, model_config = get_pretrained_model(pretrained_name)

    elif model_config is not None and model_ckpt_path is not None:
        print(f"Creating model from config")
        model = create_model_from_config(model_config)

        print(f"Loading model checkpoint from {model_ckpt_path}")
        # Load checkpoint
        copy_state_dict(model, load_ckpt_state_dict(model_ckpt_path))
        #model.load_state_dict(load_ckpt_state_dict(model_ckpt_path))

    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    if pretransform_ckpt_path is not None:
        print(f"Loading pretransform checkpoint from {pretransform_ckpt_path}")
        model.pretransform.load_state_dict(load_ckpt_state_dict(pretransform_ckpt_path), strict=True)
        print(f"Done loading pretransform")

    model.eval().requires_grad_(False).to(device)

    if model_half:
        model.to(torch.float16)
        
    print(f"Done loading model")

    return model, model_config

def clean_name(file_name: str) -> str:
    # 1. Drop file extension safely
    name = Path(file_name).stem   # removes ".wav" if present
    
    # 2. Drop leading digits, underscores, dashes
    name = name.lstrip("0123456789-_")
    
    # 3. Replace underscores with spaces
    name = name.replace("_", " ")
    
    # 4. Keep only the first part before a dash
    name = name.split("-", 1)[0]
    
    return name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    
    batch_size = args.batch_size

    # Load config from json file
    with open(args.model_config) as f:
        model_config = json.load(f)


    # append hparams to args.output_path
    args.output_path = os.path.join(args.output_path, f"bs={batch_size}_steps={args.sampling_steps}_cfg={args.cfg_scale}_median={args.median_filter_size}")
    if args.ctrl_drop != "":
        args.output_path += f"_ctrl-drop={args.ctrl_drop}"

    if 'chard' in args.sampler_type:
        args.output_path += f"_chunk-size={args.chunk_size}_stride={args.stride}_overlap-depth={args.overlap_depth}_mode={args.mode}"

    if args.diversity_test:
        args.output_path += "_diversity-test"

    # Create output dir if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)

    model, model_config = load_model(model_config, args.ckpt_path, pretransform_ckpt_path=args.pretransform_ckpt_path, model_half=args.model_half)

    # Load prompts from csv
    # They are in the "caption" column
    # Also collect ids from the "caption_id" column
    with open(args.prompt_csv_path) as csvfile:
        match args.dataset:
            case "vimsketch":
                # vimsketch csv is really just a txt file of file names, there's no header or comma delimination
                # the file naming is a bit weird though, basically we need to
                # 1. strip all leading numeric / non-alphabetic characters
                # 2. strip the file extension (.wav)
                # 3. replace underscores with spaces
                # 4. remove everything after the first hyphen
                ids_prompts = []
                for line in csvfile:
                    file_name = line.strip()
                    prompt = clean_name(file_name)
                    ids_prompts.append((file_name, None, prompt))


        # ids_prompts = [(row["caption_id"], row["track_id"], row["caption"]) for row in reader] # for song describer
        # ids_prompts = [(row["audiocap_id"], row["youtube_id"], row["caption"]) for row in reader] # for audiocaps
        #ids_prompts = [(row["CatalogID"], row["DurationSeconds"], row["Description"]) for row in reader] # For our own datasets

    # Limit to first 4000 prompts
    #ids_prompts = ids_prompts[:4000]

    # Set up accelerator
    accelerator = Accelerator()
    model = accelerator.prepare(model)



    if args.extract_features:
        # make dir
        os.makedirs(args.feature_cache_dir, exist_ok=True)
        # need to extract the features from all of the dataset
        # prompts_features = {}
        # durations = {}
        # get list of already extracted features
        fnames = copy.deepcopy(ids_prompts)
        # extracted_features = os.listdir(args.feature_cache_dir)
        # # strip .pt from extracted features
        # extracted_features = [f[:-3] for f in extracted_features]
        # fnames = [f for f in fnames if f[0][:-4] not in extracted_features]
        print(f"Extracting features for {len(fnames)} files")

        
        with accelerator.state.split_between_processes(fnames) as ids_prompts_list:

            for file_id, _, prompt in ids_prompts_list:
                # load audio
                audio_tensor, sr = torchaudio.load(f"/group2/ct/zack-novack/Vim_Sketch_Dataset/vocal_imitations/{file_id}")

                # process audio
                # To mono if needed
                if audio_tensor.ndim == 2 and audio_tensor.shape[0] > 1:
                    audio_tensor = audio_tensor.transpose(0, 1).mean(dim=0, keepdim=True)
                elif audio_tensor.ndim == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)

                # Resample if needed
                if sr != sample_rate:
                    resample_tf = T.Resample(sr, sample_rate).to(audio_tensor.device).to(audio_tensor.dtype)
                    audio_tensor = resample_tf(audio_tensor)

                duration = math.ceil(audio_tensor.shape[-1] / sample_rate)
                # durations[file_id] = duration

                # Pad/crop to input_sample_size
                if audio_tensor.shape[-1] < sample_size:
                    pad = sample_size - audio_tensor.shape[-1]
                    audio_tensor = torch.nn.functional.pad(audio_tensor, (0, pad))
                elif audio_tensor.shape[-1] > sample_size:
                    audio_tensor = audio_tensor[..., :sample_size]

                # extract features
                features = extract_features(audio_tensor.to(device), sample_rate=sample_rate)


                # prompts_features[file_id] = {k: v.cpu() for k, v in features.items()}
                sub_d = {'features': {k: v.cpu() if type(v) == torch.Tensor else v for k, v in features.items() }, 'durations': duration}
                # strip extension from file_id
                if file_id.endswith(".wav"):
                    file_id = file_id[:-4]
                torch.save(sub_d, os.path.join(args.feature_cache_dir, f"{file_id}.pt"))

            
        accelerator.wait_for_everyone()
        print(f"Done extracting features")


    # process features for current hyp
    # from stable_audio_tools.data.dataset import noncausal_median_filter
    # ids_prompts_features = {}
    # ids_prompts_features['durations'] = clean_prompts_features['durations']
    # ids_prompts_features['features'] = {}
    # for file_id, features in clean_prompts_features['features'].items():
        
    #     # Pitch: zero out low values before filtering
    #     features["pesto_pitch"][features["pesto_pitch"] < 0.1] = 0
    #     features["pesto_pitch"] = noncausal_median_filter(features["pesto_pitch"].unsqueeze(0), kernel_size=args.median_filter_size).squeeze(0)
    #     if 'pesto_pitch' in args.ctrl_drop:
    #         features["pesto_pitch"] = torch.zeros_like(features["pesto_pitch"])

    #     features["rms_energy"] = noncausal_median_filter(features["rms_energy"].unsqueeze(0), kernel_size=args.median_filter_size).squeeze(0)
    #     if 'rms_energy' in args.ctrl_drop:
    #         features["rms_energy"] = torch.ones_like(features["rms_energy"]) * (-100)

    #     features["spectral_centroid"] = noncausal_median_filter(features["spectral_centroid"].unsqueeze(0), kernel_size=args.median_filter_size).squeeze(0)
    #     if 'spectral_centroid' in args.ctrl_drop:
    #         features["spectral_centroid"] = torch.zeros_like(features["spectral_centroid"])
        
    #     ids_prompts_features['features'][file_id] = features
            

    print(f"Rendering {len(ids_prompts)} prompts...")

    # ids_prompts = [(file_id, ids_prompts_features['durations'][file_id], prompt, ids_prompts_features['features'][file_id]) for file_id, _, prompt in ids_prompts]
    t0 = time.time()
    with accelerator.state.split_between_processes(ids_prompts) as ids_prompts_list:

        # Split up ids_prompts_list by into batch_size chunks
        ids_prompts_lists = [ids_prompts_list[i:i + batch_size] for i in range(0, len(ids_prompts_list), batch_size)]

        for infos in ids_prompts_lists:
            if args.diversity_test:
                # For diversity test, we want to generate multiple samples for each prompt
                # So we will set infos to be the same prompt repeated batch_size times
                # first one will be the original prompt
                infos = [infos[0]] * batch_size
                # assert that they're all the same
                for i in range(1, batch_size):
                    assert infos[i] == infos[0], f"Prompt {i} is not the same as the first one: {infos[i]} vs {infos[0]}"
            conditioning = []


            # load in features
            clean_features = {}
            pitch_eval_ds = {}
            for i in range(len(infos)):
                file_id, _, prompt = infos[i]
                # load file
                # strip extension from file_id
                if file_id.endswith(".wav"):
                    file_id = file_id[:-4]
                clean_features[file_id] = torch.load(os.path.join(args.feature_cache_dir, f"{file_id}.pt"))
                # get pitch_eval info from features
                pitch_eval_ds[file_id] = copy.deepcopy(clean_features[file_id]['features']['eval_pitch'])
                # remove from clean_features
                del clean_features[file_id]['features']['eval_pitch']

                # smooth with median filter
                from stable_audio_tools.data.dataset import noncausal_median_filter
                features = copy.deepcopy(clean_features[file_id]['features'])
                features["pesto_pitch"][features["pesto_pitch"] < 0.1] = 0
                features["pesto_pitch"] = noncausal_median_filter(features["pesto_pitch"].unsqueeze(0), kernel_size=args.median_filter_size).squeeze(0)
                if 'pesto_pitch' in args.ctrl_drop:
                    features["pesto_pitch"] = torch.zeros_like(features["pesto_pitch"])

                features["rms_energy"] = noncausal_median_filter(features["rms_energy"].unsqueeze(0), kernel_size=args.median_filter_size).squeeze(0)
                if 'rms_energy' in args.ctrl_drop:
                    features["rms_energy"] = torch.ones_like(features["rms_energy"]) * (-100)

                features["spectral_centroid"] = noncausal_median_filter(features["spectral_centroid"].unsqueeze(0), kernel_size=args.median_filter_size).squeeze(0)
                if 'spectral_centroid' in args.ctrl_drop:
                    features["spectral_centroid"] = torch.zeros_like(features["spectral_centroid"])
                


                infos[i] = (file_id, clean_features[file_id]['durations'], prompt, features)
                print(f"Loaded features for {file_id} with duration {clean_features[file_id]['durations']} and prompt {prompt}")
                

            match args.dataset:
                # case "audiocaps":
                #     conditioning = [{"prompt": prompt, "seconds_start": 0, "seconds_total": 10} for _, _, prompt in infos]
                # case "song_describer":
                #     conditioning = [{"prompt": prompt, "seconds_start": 0, "seconds_total": min(round(float(duration)), sample_size // sample_rate)} for _, duration, prompt in infos]
                case "vimsketch":
                    if args.no_control:
                        conditioning = [{"prompt": prompt, "seconds_start": 0, "seconds_total": min(round(float(duration)), sample_size // sample_rate)} for _, duration, prompt, features in infos]
                    else:
                        conditioning = [{"prompt": prompt, "seconds_start": 0, "seconds_total": min(round(float(duration)), sample_size // sample_rate), "pesto_pitch": features["pesto_pitch"], "rms_energy": features["rms_energy"], "spectral_centroid": features["spectral_centroid"]} for _, duration, prompt, features in infos]
            

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            cond_args = {
                "model": model,
                "conditioning": conditioning,
                "steps": args.sampling_steps,
                "cfg_scale": args.cfg_scale,
                "batch_size": len(conditioning),
                "sample_size": sample_size,
                "sample_rate": sample_rate,
                "sampler_type": args.sampler_type,
                "cfg_interval": (args.cfg_interval_min, args.cfg_interval_max),
            }
            if 'chard' in args.sampler_type:
                cond_args["chunk_size"] = args.chunk_size
                cond_args["stride"] = args.stride
            if 'chard-paint' not in args.sampler_type:
                cond_args["overlap_depth"] = args.overlap_depth
                cond_args["mode"] = args.mode
            # Do the audio generation
            audio = generate_diffusion_cond(
                **cond_args
            )

            # Save the audio, one file per prompt
            for i, info in enumerate(infos):

                # Generate random id
                #id = torch.randint(0, 1000000000, (1,)).item()

                caption_id, duration, prompt, features = info
                # strip file extension from caption_id if it exists
                if caption_id.endswith(".wav"):
                    caption_id = caption_id[:-4]
                
                audio_i = audio[i].clamp(-1, 1).mul(32767).to(torch.int16)
                # audio_i = audio[i].to(torch.float32).div(torch.max(torch.abs(audio[i]))).clamp(-1, 1).mul(32767).to(torch.int16)

                # Get standard deviation of audio
                audio_std = torch.std(audio_i.float().div(32767))
                

                # extract features from generated audio to calculate error metrics
                gen_features = extract_features(audio_i.float().div(32767), sample_rate=sample_rate)
                # gen_pitch = gen_features['pesto_pitch'].to(device)
                gen_pitch = gen_features['eval_pitch']['pitch'].to(device)
                gen_energy = gen_features['rms_energy'].to(device)
                gen_centroid = gen_features['spectral_centroid'].to(device)

                # we will calculate L1 error for all
                # 1. RMS for frames where RMS > -40 dB in reference audio
                # 2. spectral centroid in semitones (i.e. x 127) for frames where RMS > -40 dB in reference audio
                # 3. pitch error in semitones (i.e. use bins_to_cents and then divide by 100) AND chroma in semitones (i.e. use bins_to_cents and then divide by 100 and then mod 12) 
                # for frames where RMS > -40 dB in reference audio AND the periodicity > 0.5 in reference audio AND generated audio
                # ref_pitch = clean_features[caption_id]['features']['pesto_pitch'].to(device)
                ref_pitch = pitch_eval_ds[caption_id]['pitch'].to(device)
                ref_energy = clean_features[caption_id]['features']['rms_energy'].to(device)
                ref_centroid = clean_features[caption_id]['features']['spectral_centroid'].to(device)
                # get duration mask
                
                ref_energy_mask = ref_energy > -40  # 21.5 is 256 / 12
                # get max probability per frame
                ref_duration_mask = torch.zeros(ref_energy_mask.size(), device=ref_energy_mask.device)
                ref_duration_mask[..., :math.ceil(duration*21.5)] = 1
                ref_energy_mask = ref_energy_mask & ref_duration_mask.bool()

                # get periodicity of pitch predictions
                ref_periodicity = periodicity(ref_pitch * 3, pitch_eval_ds[caption_id]['activations'].to(device))
                gen_periodicity = periodicity(gen_pitch * 3, gen_features['eval_pitch']['activations'].to(device))
                # upsammple ref_energy_mask to same time resolution as pitch
                pitch_ref_energy_mask = F.interpolate(ref_energy_mask.unsqueeze(0).float(), size=ref_pitch.size(-1), mode='linear', align_corners=False).squeeze(0) > 0.5
                ref_pitch_mask = ref_periodicity > 0.5
                gen_pitch_mask = gen_periodicity > 0.5
                pitch_mask = pitch_ref_energy_mask & ref_pitch_mask & gen_pitch_mask

                # convert centroids to semitones
                ref_centroid_st = ref_centroid * 127
                gen_centroid_st = gen_centroid * 127

                # convert pitches to semitones
                # ref_pitch_st = ref_pitch.argmax(dim=0, keepdim=True) / 3
                # gen_pitch_st = gen_pitch.argmax(dim=0, keepdim=True) / 3
                ref_pitch_st = ref_pitch
                gen_pitch_st = gen_pitch

                # convert pitches to chroma
                ref_pitch_chroma = ref_pitch_st % 12
                gen_pitch_chroma = gen_pitch_st % 12

                rms_error = F.l1_loss(gen_energy[ref_energy_mask], ref_energy[ref_energy_mask]).item()
                centroid_error = F.l1_loss(gen_centroid_st[ref_energy_mask], ref_centroid_st[ref_energy_mask]).item()
                pitch_error = F.l1_loss(gen_pitch_st[pitch_mask], ref_pitch_st[pitch_mask]).item()
                chroma_error = F.l1_loss(gen_pitch_chroma[pitch_mask], ref_pitch_chroma[pitch_mask]).item()

                print(f"RMS error: {rms_error}, Centroid error: {centroid_error}, Pitch error: {pitch_error}, Chroma error: {chroma_error} for prompt: {prompt}")

                error_dict = {
                    "rms_error": rms_error,
                    "centroid_error": centroid_error,
                    "pitch_error": pitch_error,
                    "chroma_error": chroma_error
                }


                if audio_std == 0:
                    print(f"STD 0 for prompt: {prompt}")

                # Cut audio_i to 10 seconds
                # duration = int(duration)
                # if duration < 285:
                #     audio_i = audio_i[:, :duration*sample_rate]
                #     print(f"Audio shape: {audio_i.shape}")
                match args.dataset:
                    # case "audiocaps":
                    #     # For audiocaps, cut to 10 seconds
                    #     audio_i = audio_i[:, :10*sample_rate]
                    # case "song_describer":
                    #     # For song describer, cut to duration seconds
                    #     duration = round(float(duration))
                    #     audio_i = audio_i[:, :min(duration*sample_rate, sample_size)]
                    case "vimsketch":
                        # For vimsketch, cut to duration seconds
                        duration = round(float(duration))
                        audio_i = audio_i[:, :min(duration*sample_rate, sample_size)].cpu()

                # strip file extension from caption_id if it exists
                if caption_id.endswith(".wav"):
                    caption_id = caption_id[:-4]
                if args.diversity_test:
                    # For diversity test, need to add i to the filename to avoid overwriting
                    caption_id = f"{caption_id}_{i}"
                # Save the audio file
                torchaudio.save(f"{args.output_path}/{caption_id}.wav", audio_i, sample_rate=sample_rate)
                with open(f"{args.output_path}/{caption_id}.json", "w") as f:
                    json.dump(error_dict, f)
                # audio_i = audio_i[:, :10*sample_rate]
                # #audio_i = audio_i[:, :2097152]

                # torchaudio.save(f"{args.output_path}/{caption_id}.wav", audio_i, sample_rate=sample_rate)
                #torchaudio.save(f"{args.output_path}/{id}.wav", audio_i, sample_rate=sample_rate)

    accelerator.wait_for_everyone()
    t1 = time.time()
    print(f"Time taken: {t1 - t0} seconds")
    print(f"Done generating audio")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run gradio interface')
    parser.add_argument('--model-config', type=str, help='Path to model config', required=False)
    parser.add_argument('--ckpt-path', type=str, help='Path to model checkpoint', required=False)
    parser.add_argument('--pretransform-ckpt-path', type=str, help='Optional to model pretransform checkpoint', required=False)
    parser.add_argument('--model-half', action='store_true', help='Whether to use half precision', required=False)
    parser.add_argument('--prompt-csv-path', type=str, help='Path to csv file with prompts', required=True)
    parser.add_argument('--output-path', type=str, help='Path to output folder', required=True)
    parser.add_argument('--batch-size', type=int, help='Batch size', required=False, default=1)
    parser.add_argument('--sampler-type', type=str, help='Sampler type', required=False, default="pingpong")
    parser.add_argument('--sampling-steps', type=int, help='Number of sampling steps', required=False, default=100)
    parser.add_argument('--cfg-scale', type=float, help='CFG scale', required=False, default=7.0)
    parser.add_argument('--dataset', type=str, help='Name of the dataset', required=False, default="audiocaps")
    parser.add_argument('--cfg-interval-min', type=float, help='CFG interval min', required=False, default=0)
    parser.add_argument('--cfg-interval-max', type=float, help='CFG interval max', required=False, default=1)
    parser.add_argument('--median-filter-size', type=int, help='Median filter size', required=False, default=1)
    parser.add_argument('--feature-cache-dir', type=str, help='Feature cache dir', required=False, default=None)
    parser.add_argument("--extract-features", action='store_true', help='Whether to extract features', required=False)
    parser.add_argument('--ctrl-drop', type=str, help='Which controls to drop', required=False, default="")
    parser.add_argument('--chunk-size', type=int, help='Chunk size for processing', required=False, default=128)
    parser.add_argument('--stride', type=int, help='Stride for processing', required=False, default=64)
    parser.add_argument('--overlap-depth', type=int, help='Overlap depth for processing', required=False, default=8)
    parser.add_argument('--no-control', action='store_true', help='Whether to run without control', required=False)
    parser.add_argument('--mode', type=str, help='chard mode', required=False, default=None)
    # seed
    parser.add_argument('--seed', type=int, help='Random seed', required=False, default=42)
    # arg to run diversity tests
    parser.add_argument('--diversity-test', action='store_true', help='Whether to run diversity tests', required=False)
    args = parser.parse_args()

    

    print("Running vimsketch")
    main(args)
    