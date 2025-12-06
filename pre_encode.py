import argparse
import gc
import json
import os
from pathlib import Path
import pickle
import random

import numpy as np
import pytorch_lightning as pl
import torch
from nnAudio.features.cqt import CQT1992v2
from torch.nn import functional as F

from stable_audio_tools.data.dataset import create_dataloader_from_config
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.pretrained import get_pretrained_model
from stable_audio_tools.models.utils import load_ckpt_state_dict, copy_state_dict
from stable_audio_tools.models.diffusion import ConditionedDiffusionModelWrapper
import torchaudio


def load_model(model_config=None, model_ckpt_path=None, pretrained_name=None, model_half=False):
    if pretrained_name is not None:
        print(f"Loading pretrained model {pretrained_name}")
        model, model_config = get_pretrained_model(pretrained_name)

    elif model_config is not None and model_ckpt_path is not None:
        print(f"Creating model from config")
        model = create_model_from_config(model_config)

        print(f"Loading model checkpoint from {model_ckpt_path}")
        copy_state_dict(model, load_ckpt_state_dict(model_ckpt_path))

    model.eval().requires_grad_(False)

    if model_half:
        model.to(torch.float16)

    print("Done loading model")

    return model, model_config


def causal_rms_energy(audio, sr=48000):
    # audio: (B, C, T) or (C, T)
    if audio.ndim == 2:
        audio = audio.unsqueeze(0)
    B, C, T = audio.shape

    # Mixdown to mono for loudness
    audio_mono = audio.mean(dim=1)  # (B, T)

    # Set causal STFT parameters
    n_fft = 800
    hop_length = 800
    # Causal window: rectangular, only looks at current and past samples
    window = torch.ones(n_fft, device=audio.device)
    

    # Compute STFT with center=False for causality
    spec = torch.stft(
        audio_mono, n_fft=n_fft, hop_length=hop_length, window=window,
        return_complex=True, center=False
    )  # (B, F, frames)
    mag = spec.abs()  # (B, F, frames)

    # Get A-weighting curve for the frequencies
    freqs = torch.fft.rfftfreq(n_fft, 1.0 / sr).to(audio.device)  # (F,)

    def a_weighting(f):
        f_sq = f ** 2
        ra = (f_sq + 20.6 ** 2) * (f_sq + 12200 ** 2)
        num = (12200 ** 2) * (f_sq) ** 2
        den = ra * torch.sqrt((f_sq + 107.7 ** 2) * (f_sq + 737.9 ** 2))
        a = num / (den + 1e-20)
        a_db = 2.0 + 20 * torch.log10(a + 1e-20)
        return a_db

    a_weight = a_weighting(freqs)  # (F,)
    a_weight_lin = 10 ** (a_weight / 20)  # convert dB to linear

    # Apply A-weighting to magnitude spectrogram
    mag_weighted = mag * a_weight_lin[None, :, None]  # (B, F, frames)

    # Sum across frequency bins and compute RMS per frame
    loudness = torch.sqrt((mag_weighted ** 2).sum(dim=1) / mag_weighted.shape[1])  # (B, frames)
    # Convert to dB
    loudness_db = 20 * torch.log10(loudness + 1e-20)  # (B, frames)
    return loudness_db

def rms_energy_noncausal(audio, sr=48000):
    # audio: (B, C, T) or (C, T)
    if audio.ndim == 2:
        audio = audio.unsqueeze(0)
    B, C, T = audio.shape

    # Mixdown to mono for loudness
    audio_mono = audio.mean(dim=1)  # (B, T)

    # Non-causal STFT parameters
    n_fft = 1600
    hop_length = 1600
    # Use Hann window for non-causal (centered) STFT
    window = torch.hann_window(n_fft, device=audio.device)

    # Compute STFT with center=True for non-causal
    spec = torch.stft(
        audio_mono, n_fft=n_fft, hop_length=hop_length, window=window,
        return_complex=True, center=True
    )  # (B, F, frames)
    mag = spec.abs()  # (B, F, frames)

    # Get A-weighting curve for the frequencies
    freqs = torch.fft.rfftfreq(n_fft, 1.0 / sr).to(audio.device)  # (F,)

    def a_weighting(f):
        f_sq = f ** 2
        ra = (f_sq + 20.6 ** 2) * (f_sq + 12200 ** 2)
        num = (12200 ** 2) * (f_sq) ** 2
        den = ra * torch.sqrt((f_sq + 107.7 ** 2) * (f_sq + 737.9 ** 2))
        a = num / (den + 1e-20)
        a_db = 2.0 + 20 * torch.log10(a + 1e-20)
        return a_db

    a_weight = a_weighting(freqs)  # (F,)
    a_weight_lin = 10 ** (a_weight / 20)  # convert dB to linear

    # Apply A-weighting to magnitude spectrogram
    mag_weighted = mag * a_weight_lin[None, :, None]  # (B, F, frames)

    # Sum across frequency bins and compute RMS per frame
    loudness = torch.sqrt((mag_weighted ** 2).sum(dim=1) / mag_weighted.shape[1])  # (B, frames)
    # Convert to dB
    loudness_db = 20 * torch.log10(loudness + 1e-20)  # (B, frames)
    return loudness_db

def rms_energy_noncausal_256(audio, sr=44100):
    # audio: (B, C, T) or (C, T)
    if audio.ndim == 2:
        audio = audio.unsqueeze(0)
    B, C, T = audio.shape

    # Mixdown to mono for loudness
    audio_mono = audio.mean(dim=1)  # (B, T)

    # Non-causal STFT parameters
    n_fft = 2048
    hop_length = 2049
    # Use Hann window for non-causal (centered) STFT
    window = torch.hann_window(n_fft, device=audio.device)

    # Compute STFT with center=True for non-causal
    spec = torch.stft(
        audio_mono, n_fft=n_fft, hop_length=hop_length, window=window,
        return_complex=True, center=True
    )  # (B, F, frames)
    mag = spec.abs()  # (B, F, frames)

    # Get A-weighting curve for the frequencies
    freqs = torch.fft.rfftfreq(n_fft, 1.0 / sr).to(audio.device)  # (F,)

    def a_weighting(f):
        f_sq = f ** 2
        ra = (f_sq + 20.6 ** 2) * (f_sq + 12200 ** 2)
        num = (12200 ** 2) * (f_sq) ** 2
        den = ra * torch.sqrt((f_sq + 107.7 ** 2) * (f_sq + 737.9 ** 2))
        a = num / (den + 1e-20)
        a_db = 2.0 + 20 * torch.log10(a + 1e-20)
        return a_db

    a_weight = a_weighting(freqs)  # (F,)
    a_weight_lin = 10 ** (a_weight / 20)  # convert dB to linear

    # Apply A-weighting to magnitude spectrogram
    mag_weighted = mag * a_weight_lin[None, :, None]  # (B, F, frames)

    # Sum across frequency bins and compute RMS per frame
    loudness = torch.sqrt((mag_weighted ** 2).sum(dim=1) / mag_weighted.shape[1])  # (B, frames)
    # Convert to dB
    loudness_db = 20 * torch.log10(loudness + 1e-20)  # (B, frames)
    return loudness_db
import librosa
import warnings
import scipy
MIN_DB = -100.
REF_DB = 20.
CREPE_SR = 16000
CREPE_WIN_SIZE = 1024

def get_perceptual_weights_cpu():
    """Compute A-weighting curve once using librosa (on CPU)"""
    freqs = librosa.fft_frequencies(sr=CREPE_SR, n_fft=CREPE_WIN_SIZE)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        a_weight = librosa.A_weighting(freqs) - REF_DB
    return torch.tensor(a_weight, dtype=torch.float32)  # shape (F,)

def perceptual_weights():
    """A-weighted frequency-dependent perceptual loudness weights"""
    frequencies = librosa.fft_frequencies(sr=CREPE_SR,
                                          n_fft=CREPE_WIN_SIZE)

    # A warning is raised for nearly inaudible frequencies, but it ends up
    # defaulting to -100 db. That default is fine for our purposes.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return librosa.A_weighting(frequencies)[:, None] - REF_DB


# Cache CPU-computed weights (computed once)
_A_WEIGHT_CPU = get_perceptual_weights_cpu()[:, None]  # shape (F, 1)
hann_window = torch.tensor(scipy.signal.get_window('hann', CREPE_WIN_SIZE, fftbins=True))


def a_weighted_gpu(audio: torch.Tensor, sample_rate: int=44100, hop_length=None, pad=True) -> torch.Tensor:
    """
    GPU-based A-weighted loudness estimation using torchaudio and precomputed perceptual weights.
    
    Args:
        audio: Tensor of shape (B, C, T), on GPU
        sample_rate: Sampling rate of audio
        hop_length: Hop size (defaults to 10ms)
        pad: Whether to center the STFT
    
    Returns:
        Tensor of shape (1, n_frames) representing loudness in dB
    """

    if not hasattr(a_weighted_gpu, "weights"):
        a_weighted_gpu.weights = _A_WEIGHT_CPU.to(audio.device)
        a_weighted_gpu.window = hann_window.to(audio.device)
    

    if audio.ndim == 3:
        audio = audio.mean(dim=1)
    if audio.ndim == 2 and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)  # (1, T)
    audio = audio.to(torch.float32)  
    device = audio.device
    hop_length = sample_rate // 100 if hop_length is None else hop_length

    # Resample if needed
    if sample_rate != CREPE_SR:
        audio = torchaudio.functional.resample(
            waveform=audio,
            orig_freq=sample_rate,
            new_freq=CREPE_SR
        )
        hop_length = int(hop_length * CREPE_SR / sample_rate)
        sample_rate = CREPE_SR

    # Compute STFT
    stft = torch.stft(audio,
                      n_fft=CREPE_WIN_SIZE,
                      hop_length=hop_length,
                      win_length=CREPE_WIN_SIZE,
                      window=a_weighted_gpu.window,
                      center=pad,
                      normalized=False,
                      pad_mode='constant',
                      return_complex=True)  # (B, F, T)
    mag = stft.abs().pow(2)

    # Convert to dB
    db = 10.0 * torch.log10(torch.maximum(torch.tensor(1e-10), mag))
    db = torch.maximum(db, db.max() - 80)

    # Apply precomputed perceptual weights
    weights = a_weighted_gpu.weights
    weighted_db = db + weights

    # Threshold
    weighted_db = torch.clamp(weighted_db, min=MIN_DB)

    # Mean across frequency bins
    loudness = weighted_db.mean(dim=1, keepdim=True)  # (B, 1, T)
    return loudness



from pesto import load_model as pesto_load_model

def pesto_pitch_600(audio, sr=48000, hop_length=3000):
    # pesto expects (B, T) in float32, mono
    if audio.ndim == 3 and audio.shape[1] > 1:
        audio = audio.mean(dim=1)  # (B, T)
    elif audio.ndim == 2 and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    # if getattr(pesto_pitch, "model", None) is None:
    with torch.no_grad():
        if not hasattr(pesto_pitch, "model"):
            print("Loading pesto pitch extraction model")
            # Load the pesto model
            sr = int(sr)
            pesto_pitch.model = pesto_load_model("mir-1k_g7", step_size=16.68, sampling_rate=sr).to(audio.device)
        model = pesto_pitch.model
        # pesto expects shape (B, T)
        _, _2, _3, activations = model(audio, sr=sr)
        return activations  # shape: (B, frames, num_classes)

def pesto_pitch_300(audio, sr=48000, hop_length=3000):
    # pesto expects (B, T) in float32, mono
    if audio.ndim == 3 and audio.shape[1] > 1:
        audio = audio.mean(dim=1)  # (B, T)
    elif audio.ndim == 2 and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    # if getattr(pesto_pitch, "model", None) is None:
    with torch.no_grad():
        if not hasattr(pesto_pitch, "model"):
            print("Loading pesto pitch extraction model")
            # Load the pesto model
            sr = int(sr)
            pesto_pitch.model = pesto_load_model("mir-1k_g7", step_size=31.1, sampling_rate=sr).to(audio.device)
        model = pesto_pitch.model
        # pesto expects shape (B, T)
        _, _2, _3, activations = model(audio, sr=sr)
        return activations  # shape: (B, frames, num_classes)


def pesto_pitch_256(audio, sr=44100, hop_length=3000, return_predictions=False):
    # pesto expects (B, T) in float32, mono
    if audio.ndim == 3 and audio.shape[1] > 1:
        audio = audio.mean(dim=1)  # (B, T)
    elif audio.ndim == 2 and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    # if getattr(pesto_pitch, "model", None) is None:
    with torch.no_grad():
        if not hasattr(pesto_pitch_256, "model"):
            print("Loading pesto pitch extraction model")
            # Load the pesto model
            sr = int(sr)
            pesto_pitch_256.model = pesto_load_model("mir-1k_g7", step_size=20, sampling_rate=sr).to(audio.device)
        model = pesto_pitch_256.model
        # pesto expects shape (B, T)
        preds, _2, _3, activations = model(audio, sr=sr)
        if return_predictions:
            return preds, activations.transpose(-1, -2)
        return activations.transpose(-1, -2)  # shape: (B, frames, num_classes)

# def spectral_centroid(audio, sr=44100, n_fft=2048, hop_length=512):
#     # audio: (B, C, T) or (C, T)
#     if audio.ndim == 2:
#         audio = audio.unsqueeze(0)
#     B, C, T = audio.shape

#     # Mixdown to mono
#     audio_mono = audio.mean(dim=1)  # (B, T)

#     # Compute STFT
#     window = torch.hann_window(n_fft, device=audio.device)
#     spec = torch.stft(audio_mono, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)  # (B, F, frames)
#     mag = spec.abs()  # (B, F, frames)

#     # Frequency bins (Hz)
#     freqs = torch.fft.rfftfreq(n_fft, 1.0 / sr).to(audio.device)  # (F,)

#     # Spectral centroid: sum(freq * mag) / sum(mag)
#     num = (mag * freqs[None, :, None]).sum(dim=1)
#     den = mag.sum(dim=1) + 1e-10
#     centroid_hz = num / den  # (B, frames)

#     # Convert Hz to MIDI-like (continuous)
#     centroid_midi = 69 + 12 * torch.log2(centroid_hz / 440.0 + 1e-10)
#     centroid_midi = torch.clamp(centroid_midi, min=0, max=127)

#     # Scale to (0, 1) by dividing by 127 (G9)
#     centroid_scaled = centroid_midi / 127.0

#     return centroid_scaled  # (B, frames)


def spectral_centroid(audio, sr=44100, n_fft=2048, hop_length=512):
    # Compute STFT frequencies
    if audio.ndim == 2:
        audio = audio.unsqueeze(0)
    B, C, T = audio.shape

    # Mixdown to mono
    audio_mono = audio.mean(dim=1)  # (B, T)

    # Compute STFT
    window = torch.hann_window(n_fft, device=audio.device)
    spectrogram = torch.stft(audio_mono, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)  # (B, F, frames)
    spectrogram = spectrogram.abs()  # (B, F, frames)

    frequencies = torch.abs(torch.fft.fftfreq(
        2 * (spectrogram.shape[1] - 1),
        1 / sr,
        device=spectrogram.device
    )[:spectrogram.shape[1]])

    # Compute centroid
    centroid = (
        frequencies * spectrogram.transpose(1, 2)
    ).sum(dim=2) / spectrogram.sum(dim=1).squeeze()
    centroid_midi = 69 + 12 * torch.log2(centroid / 440.0 + 1e-10)
    centroid_midi = torch.clamp(centroid_midi, min=0, max=127)
    # Scale to (0, 1) by dividing by 127 (G9)
    centroid_scaled = centroid_midi / 127.0
    return centroid_scaled  # (frames,)



def cqt_topk(audio, sr=44100, hop_length=512, fmin=8.18, fmax=12543.85, bins_per_octave=12, top_k=20):
    # Compute CQT
    if getattr(cqt_topk, "cqt", None) is None:
        cqt_topk.cqt = CQT1992v2(
            sr=sr,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax,
            bins_per_octave=bins_per_octave,
        ).to(audio.device)
    # if audio is stereo, average to mono
    if audio.shape[1] == 2:
        audio = audio.mean(dim=1, keepdim=True)
    cqt_spec = cqt_topk.cqt(audio)  # (B, F, T)
    # Get top-k frequencies for each time step
    topk_freqs, topk_indices = torch.topk(cqt_spec, k=top_k, dim=1)  # (B, top_k, T)
    # return top-k indices
    return topk_indices



        

class PreEncodedLatentsInferenceWrapper(pl.LightningModule):
    def __init__(
        self, 
        model,
        output_path,
        is_discrete=False,
        model_half=False,
        model_config=None,
        dataset_config=None,
        sample_size=1920000,
        args_dict=None,
        reparse_controls=False,
        controls=[]
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.output_path = Path(output_path)
        self.controls = controls
        self.reparse_controls = reparse_controls

    def prepare_data(self):
        # runs on rank 0
        self.output_path.mkdir(parents=True, exist_ok=True)
        details_path = self.output_path / "details.json"
        if not details_path.exists():  # Only save if it doesn't exist
            details = {
                "model_config": self.hparams.model_config,
                "dataset_config": self.hparams.dataset_config,
                "sample_size": self.hparams.sample_size,
                "args": self.hparams.args_dict
            }
            details_path.write_text(json.dumps(details))

    def setup(self, stage=None):
        # runs on each device
        process_dir = self.output_path
        process_dir.mkdir(parents=True, exist_ok=True)
        # read through outputh_path and check which files already exist
        existing_files = set()
        for file in process_dir.glob("*.npy"):
            existing_files.add(file.stem)
        self.existing_files = existing_files

    def validation_step(self, batch, batch_idx):
        audio, metadata = batch 

        # if all the files have already been processed, skip
        if all(f"{md['relpath'].split('.')[0]}" in self.existing_files for i, md in enumerate(metadata)):
            # check if any of the control files exist, if they don't, extract just the controls from the 
            # audio, as the latent is already extracted
            if len(self.controls) > 0:
                for i, md in enumerate(metadata):
                    latent_id = md['relpath'].split(".")[0] if 'relpath' in md else f"{self.global_rank:03d}{batch_idx:06d}{i:04d}"
                    latent_id = latent_id.replace("/", "_").replace("\\", "_")
                    control_path = self.output_path / f"{latent_id}_controls.pkl"
                    if not control_path.exists() or self.reparse_controls:
                        # extract controls from audio
                        controls = {control: eval(control)(audio[i]).cpu() for control in self.controls}
                        for control_name, control_value in controls.items():
                            control_value = control_value.squeeze()  # add batch dim
                            if control_value.ndim == 1:
                                # add 2 dims
                                control_value = control_value.unsqueeze(0).unsqueeze(0)
                            elif control_value.ndim == 2:
                                # add channel dim
                                control_value = control_value.unsqueeze(0)
                            controls[control_name] = F.interpolate(control_value, size=256, mode="linear").squeeze()
                            # if the control is 1D, add a channel dim
                            if controls[control_name].ndim == 1:
                                controls[control_name] = controls[control_name].unsqueeze(0)
                            # print(f"Extracted control {control_name} with shape {(controls[control_name].shape, control_value.shape)} for {latent_id}")
                        with open(control_path, "wb") as f:
                            pickle.dump(controls, f)

            return

        if audio.ndim == 4 and audio.shape[0] == 1:
            audio = audio[0]

        # if audio is not stereo, and the autoencoder expects stereo, convert to stereo by repeating the channel 
        if audio.ndim == 3 and audio.shape[1] == 1 and self.model.io_channels == 2:
            print("Converting mono audio to stereo by repeating the channel")
            audio = audio.repeat(1, 2, 1)
            

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if self.hparams.model_half:
            audio = audio.to(torch.float16)

        with torch.no_grad():
            if not self.hparams.is_discrete:
                latents = self.model.encode(audio)
            else:
                _, info = self.model.encode(audio, return_info=True)
                latents = info[self.model.bottleneck.tokens_id]

        latents = latents.cpu().numpy()

        # extract extra features
        # lets do RMS energy, pesto pitch extraction, spectral centroid, and CQT

        # # first, RMS energy
        # loudness_db = rms_energy(audio)
        # loudness_db = loudness_db.cpu().numpy()



        # Save each sample in the batch
        for i, latent in enumerate(latents):
            md = metadata[i]
            latent_id = md['relpath'].split(".")[0] if 'relpath' in md else f"{self.global_rank:03d}{batch_idx:06d}{i:04d}"
            # if latent_id has any subdirectories, just convert into a flat name
            latent_id = latent_id.replace("/", "_").replace("\\", "_")

            # Save latent as numpy file
            latent_path = self.output_path / f"{latent_id}.npy"
            if self.hparams.args_dict.get("memmap", False):
                # Save as memmap if specified
                latent_path = latent_path.with_suffix('.mmap')
                memmap_latent = np.memmap(latent_path, dtype='float32', mode='w+', shape=latent.shape)
                memmap_latent[:] = latent
                memmap_latent.flush()
            else:
                with open(latent_path, "wb") as f:
                    np.save(f, latent)

            
            padding_mask = F.interpolate(
                md["padding_mask"].unsqueeze(0).unsqueeze(1).float(),
                size=latent.shape[1],
                mode="nearest"
            ).squeeze().int()
            md["padding_mask"] = padding_mask.cpu().numpy().tolist()

            # Convert tensors in md to serializable types
            for k, v in md.items():
                if isinstance(v, torch.Tensor):
                    md[k] = v.cpu().numpy().tolist()

            # Save metadata to json file
            metadata_path = self.output_path / f"{latent_id}.json"
            with open(metadata_path, "w") as f:
                json.dump(md, f)

            # Save controls if specified
            if len(self.controls) > 0:
                control_path = self.output_path / f"{latent_id}_controls.pkl"
                if not control_path.exists():
                    # extract controls from audio
                    controls = {control: eval(control)(audio[i]).cpu() for control in self.controls}
                    for control_name, control_value in controls.items():
                        control_value = control_value.squeeze()  # add batch dim
                        if control_value.ndim == 1:
                            # add 2 dims
                            control_value = control_value.unsqueeze(0).unsqueeze(0)
                        elif control_value.ndim == 2:
                            # add channel dim
                            control_value = control_value.unsqueeze(0)
                        controls[control_name] = F.interpolate(control_value, size=256, mode="linear").squeeze()
                        # if the control is 1D, add a channel dim
                        if controls[control_name].ndim == 1:
                            controls[control_name] = controls[control_name].unsqueeze(0)
                        # print(f"Extracted control {control_name} with shape {(controls[control_name].shape, control_value.shape)} for {latent_id}")
                    with open(control_path, "wb") as f:
                        pickle.dump(controls, f)

    def shard_and_tar(self):
        # after data is processed, divide it into shards and tar them
        # only do this on rank 0
        if self.global_rank != 0:
            print("Skipping sharding and tarring on non-rank 0 processes.")
            return
        import webdataset as wds
        from tqdm import tqdm

        # make new output path for shards
        shard_output_path = self.output_path / "shards"
        shard_output_path.mkdir(parents=True, exist_ok=True)
        # get all npy files in output_path
        npy_files = list(self.output_path.glob("*.npy"))
        if not npy_files:
            print("No npy files found in output path, skipping sharding.")
            return
        # sort files by name
        npy_files.sort(key=lambda x: x.name)
        # group files into shards of roughly 1GB each
        with wds.ShardWriter(
            str(shard_output_path / "shard-%06d.tar"),
            maxsize=1e9,  # 1GB per shard
        ) as sink:
            for npy_file in tqdm(npy_files, desc="Sharding files"):
                # read the latent
                latent = np.load(npy_file, allow_pickle=True)
                # read the metadata
                metadata_path = npy_file.with_suffix('.json')
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                # create a sample for webdataset
                sample = {
                    "__key__": npy_file.stem,
                    "latent.npy": latent,
                    "metadata.json": metadata
                }
                # check for controls
                control_path = npy_file.parent / (npy_file.stem + '_controls.pkl')
                if control_path.exists():
                    with open(control_path, "rb") as f:
                        controls = pickle.load(f)
                    # add controls to sample
                    for control_name, control_value in controls.items():
                        sample[f"{control_name}.pickle"] = control_value

                # write the sample to the shard
                sink.write(sample)

                # remove the original files
                # os.remove(npy_file)
                # os.remove(metadata_path)
        print(f"Sharded files saved to {shard_output_path}.")
            



    def configure_optimizers(self):
        return None


def main(args):
    with open(args.model_config) as f:
        model_config = json.load(f)

    with open(args.dataset_config) as f:
        dataset_config = json.load(f)


    # select correct pitch and RMS functions based on noncausal flag and model sample rate
    global rms_energy, pesto_pitch
    if args.noncausal:
        if model_config["sample_rate"] == 44100:
            rms_energy = a_weighted_gpu
            pesto_pitch = pesto_pitch_256
        else:
            rms_energy = rms_energy_noncausal
            pesto_pitch = pesto_pitch_300
    else:
        rms_energy = causal_rms_energy
        pesto_pitch = pesto_pitch_600

    model, model_config = load_model(
        model_config=model_config,
        model_ckpt_path=args.ckpt_path,
        model_half=args.model_half
    )
    if type(model) == ConditionedDiffusionModelWrapper:
        # model is diffusion model, just get its autoencoder
        model = model.pretransform

    data_loader = create_dataloader_from_config(
        dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=args.sample_size,
        audio_channels=model_config.get("audio_channels", 2),
        shuffle=args.shuffle
    )

    blacklist_pth = "/group2/mfm/data/WavCaps/json_files/blacklist/blacklist_exclude_test_ac.json"
    with open(blacklist_pth, "r") as f:
        blacklist = json.load(f)
        blacklist_as = blacklist['AudioSet']
        blacklist_as = set([v.split(".")[0] for v in blacklist_as])  # remove file extensions
        blacklist_fs = blacklist['FreeSound']
        blacklist_fs = set([v.split(".")[0] for v in blacklist_fs])  # remove file extensions
        old_len = len(data_loader.dataset.filenames)
        def is_blacklisted(filename):
            if "AudioSet" in filename:
                return filename.split("/")[-1].split(".")[0] in blacklist_as
            elif "FreeSound" in filename:
                return filename.split("/")[-1].split(".")[0] in blacklist_fs
            return False

        data_loader.dataset.filenames = [f for f in data_loader.dataset.filenames if not is_blacklisted(f)]
        print(f"Filtered out {old_len - len(data_loader.dataset.filenames)} blacklisted files from the dataset. Now contains {len(data_loader.dataset.filenames)} files.")


    if args.filter_csv_pth is not None:
        import pandas as pd
        from os import path
        print(f"Filtering dataset using {args.filter_csv_pth}")
        df = pd.read_csv(args.filter_csv_pth)
        # convert to set
        filter_set = set(df["filename"].tolist())
        data_loader.dataset.filenames = [f for f in data_loader.dataset.filenames if path.basename(f) in filter_set]
        print(f"Dataset contains {len(data_loader.dataset.filenames)} files after filtering.")


    pl_module = PreEncodedLatentsInferenceWrapper(
        model=model,
        output_path=args.output_path,
        is_discrete=args.is_discrete,
        model_half=args.model_half,
        model_config=args.model_config,
        dataset_config=args.dataset_config,
        sample_size=args.sample_size,
        args_dict=vars(args),
        reparse_controls=args.reparse_controls,
        controls=dataset_config.get("controls", [])
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        num_nodes = args.num_nodes,
        strategy=args.strategy,
        precision="16-true" if args.model_half else "32",
        max_steps=args.limit_batches if args.limit_batches else -1,
        logger=False,  # Disable logging since we're just doing inference
        enable_checkpointing=False,
    )
    trainer.validate(pl_module, data_loader)
    if not args.noshard:
        pl_module.shard_and_tar()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Encode audio dataset to VAE latents using PyTorch Lightning')
    parser.add_argument('--model-config', type=str, help='Path to model config', required=False)
    parser.add_argument('--ckpt-path', type=str, help='Path to unwrapped autoencoder model checkpoint', required=False)
    parser.add_argument('--model-half', action='store_true', help='Whether to use half precision')
    parser.add_argument('--dataset-config', type=str, help='Path to dataset config file', required=True)
    parser.add_argument('--output-path', type=str, help='Path to output folder', required=True)
    parser.add_argument('--batch-size', type=int, help='Batch size', default=1)
    parser.add_argument('--sample-size', type=int, help='Number of audio samples to pad/crop to', default=1320960)
    parser.add_argument('--is-discrete', action='store_true', help='Whether the model is discrete')
    parser.add_argument('--num-nodes', type=int, help='Number of GPU nodes', default=1)
    parser.add_argument('--num-workers', type=int, help='Number of dataloader workers', default=4)
    parser.add_argument('--strategy', type=str, help='PyTorch Lightning strategy', default='auto')
    parser.add_argument('--limit-batches', type=int, help='Limit number of batches (optional)', default=None)
    parser.add_argument('--shuffle', action='store_true', help='Shuffle dataset')
    parser.add_argument('--memmap', action='store_true', help='Use memmap for saving latents')
    parser.add_argument('--filter-csv-pth', type=str, help='Path to CSV file with filenames to filter dataset', default=None)
    parser.add_argument('--noncausal', action='store_true', help='Use non-causal STFT for RMS energy calculation')
    parser.add_argument('--noshard', action='store_true', help='Do not shard the output files into tar files')
    parser.add_argument('--reparse-controls', action='store_true', help='Reparse controls from audio even if they already exist')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility', default=42)
    args = parser.parse_args()
    # set all random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


    main(args)