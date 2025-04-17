import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import argparse
from tqdm import tqdm
from pathlib import Path
from feature_extractors import ffmpeg_extract, save_faces, save_faces_with_gender
import pandas as pd

def main(args):
    dataset_dir = Path(args.dataset_dir)
    csv_path = dataset_dir / "Processed" / "my_iemocap.csv"
    df = pd.read_csv(csv_path)
    new_data = []   
    total_items = len(df)
    pbar = tqdm(total=total_items, desc="Processing")
    for i in range(total_items):
        audio_path = df['audio_path'][i]
        entry_name = os.path.splitext(os.path.basename(audio_path))[0]
        video_path = dataset_dir / "subvideo" /f'{entry_name}.mp4'
        assert os.path.exists(video_path), f"Video file 'subvideo/{entry_name}.mp4' does not exist!"

        keyframe_dir = Path(args.dataset_dir) / "V" / f"{entry_name}.jpg"
        keyframe_dir.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_extract(video_path, str(keyframe_dir), mode='image')
        

        face_dir = Path(args.dataset_dir) / "Face" / f"{entry_name}"
        face_dir.parent.mkdir(parents=True, exist_ok=True)
        save_faces_with_gender(keyframe_dir, face_dir)

        new_data.append({
            'dialog_id': df['dialog_id'][i],
            'utterance_id': df['utterance_id'][i],
            'speaker': df['speaker'][i],
            'start_time': df['start_time'][i],
            'end_time': df['end_time'][i],
            'audio_path': audio_path,
            'video_path': f"subvideo/{entry_name}.mp4",
            'image_path': f"V/{entry_name}.jpg",
            'face_dir': f"Face/{entry_name}",
            'emotion': df['emotion'][i],
            'vad': df['vad'][i],
            'text': df['text'][i]
        })

        pbar.update(1)
        pbar.set_description(f"Processing {entry_name}")
    pbar.close()    
    new_df = pd.DataFrame(new_data)

    new_csv_path = dataset_dir / "Processed" / "processed_data.csv"
    new_df.to_csv(new_csv_path, index=False)
    print(f"New data saved to {new_csv_path}")   


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data")
    parser.add_argument("--dataset_root_dir", type=str, default="datasets/ERC", help="Root directory of datasets")
    parser.add_argument("--dataset_dir", type=str, default="datasets/ERC/IEMOCAP/", help="Path to dataset folder")
    parser.add_argument("--dataset_name", type=str, default="IEMOCAP", help="Name of the dataset")
    args = parser.parse_args()
    main(args)