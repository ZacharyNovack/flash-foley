import pandas as pd
import random
import json

def extract_id_caption_d(d):
    # dictionary has "data" field, which is a list of dictionaries
    # each with a lot of info, but importantly an "id" and "caption" field
    return pd.DataFrame({
        "id": [item["id"].split(".")[0] for item in d["data"]],
        "caption": [item["caption"] for item in d["data"]]
    })

def get_custom_metadata(info, audio):
    
    # check if function (as object) contains csv, if not read it in
    if not hasattr(get_custom_metadata, "csv_data"):
        # Load the json
        with open("/path/to/as_final.json", "r") as f:
            audioset_data = extract_id_caption_d(json.load(f))
        with open("/path/to/bbc_final.json", "r") as f:
            bbc_data = extract_id_caption_d(json.load(f))
        with open("/path/to/fsd_final.json", "r") as f:
            freesound_data = extract_id_caption_d(json.load(f))
        with open("/path/to/sb_final.json", "r") as f:
            soundbible_data = extract_id_caption_d(json.load(f))
        # Concatenate all dataframes
        audioset_data['source'] = 'AudioSet'
        bbc_data['source'] = 'BBC'
        freesound_data['source'] = 'FreeSound'
        soundbible_data['source'] = 'SoundBible'

        csv_data = {}
        for df in [audioset_data, bbc_data, freesound_data, soundbible_data]:
            df['filename'] = df['id']
            source = df['source'].iloc[0]
            df = df[["filename", "caption"]]
            csv_data[source] = df.set_index("filename").T.to_dict()

        # convert to dictionary of audio_filename: {caption1, caption2} pairs
        get_custom_metadata.csv_data = csv_data

    def get_metadata_for_audio(audio_info):
        filename = audio_info["path"]
        for source, data in get_custom_metadata.csv_data.items():
            if source in filename:
                file_id = filename.split("/")[-1].split(".")[0]
                return data.get(file_id, {"caption": ""})
    # Get the metadata for the audio file
    metadata = get_metadata_for_audio(info)

    
    caption = metadata["caption"]


    return {"prompt": caption}