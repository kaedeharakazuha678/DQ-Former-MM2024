import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import argparse
from tqdm import tqdm
from pathlib import Path
from feature_extractors import ffmpeg_extract, save_faces
from utils.io_utils import read_yaml, save_result


def main(args):
    data = read_yaml(os.path.join(args.dataset_dir, "datasets.yaml"))
    total_items = sum(len(entries) for entries in data.values())
    pbar = tqdm(total=total_items, desc="Processing")
    mp4data = {}
    for dataset_type, entries in data.items(): # dataset_type: dev, train, test
        if dataset_type not in mp4data:
            mp4data[dataset_type] = {}

        for entry_name, entry_data in tqdm(entries.items(), desc=f"Processing {dataset_type}"): # entry_name: dia0_utt0, dia0_utt1, ...
            video_path = os.path.join(args.dataset_dir, dataset_type, f"{entry_name}.mp4") 
            assert os.path.exists(video_path), f"Video file {dataset_type}/{entry_name}.mp4 does not exist!"   
            # extract keyframe
            keyframe_dir = Path(args.dataset_dir) / "V" / dataset_type / f"{entry_name}.jpg"
            keyframe_dir.parent.mkdir(parents=True, exist_ok=True)
            ffmpeg_extract(video_path, str(keyframe_dir), mode='image')

            # extract speaker face
            face_dir = Path(args.dataset_dir) / "Face" / dataset_type / f"{entry_name}"
            face_dir.parent.mkdir(parents=True, exist_ok=True)
            save_faces(keyframe_dir, face_dir)
            
            # extract audio 
            audio_path = Path(args.dataset_dir) / "A" / dataset_type / f"{entry_name}.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            ffmpeg_extract(video_path, str(audio_path))

            mp4data[dataset_type][entry_name] = {
                "video_path": f"{dataset_type}/{entry_name}.mp4",
                "audio_path": f"A/{dataset_type}/{entry_name}.wav",
                "image_path": f"V/{dataset_type}/{entry_name}.jpg",
                "face_dir": f"Face/{dataset_type}/{entry_name}",
                "text": entry_data["Utterance"],
                "emotion": entry_data["Emotion"],
                "sentiment": entry_data["Sentiment"],
                "speaker": entry_data["Speaker"],
                "utterance_id": entry_data["Utterance_ID"],
                "dialogue_id": entry_data["Dialogue_ID"],
            }

            pbar.update(1)
            pbar.set_description(f"Processing {dataset_type}/{entry_name}")
        out_dir = f"{args.dataset_dir}/Processed/{dataset_type}_data.pkl"
        save_result(mp4data[dataset_type], out_dir)
    save_dir = f"{args.dataset_dir}/Processed/{args.dataset_name}_data.pkl"
    save_result(mp4data, save_dir)
    pbar.close()              


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data")
    parser.add_argument("--dataset_root_dir", type=str, default="datasets/ERC", help="Root directory of datasets")
    parser.add_argument("--dataset_dir", type=str, default="datasets/ERC/MELD/", help="Path to dataset folder")
    parser.add_argument("--dataset_name", type=str, default="MELD", help="Name of the dataset")
    args = parser.parse_args()
    main(args)