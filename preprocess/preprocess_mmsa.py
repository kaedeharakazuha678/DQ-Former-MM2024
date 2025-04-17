import json
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm 
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.io_utils import save_result
from feature_extractors import ffmpeg_extract, save_faces

def read_label_file(dataset_name, dataset_root_dir, dataset_dir):
    '''
    partly copy from https://github.com/thuiar/MMSA-FET.git
    '''
    # Locate and read label.csv file, you can use dataset_name + dataset_root_dir of dataset_dir
    assert dataset_name is not None or dataset_dir is not None, "Either 'dataset_name' or 'dataset_dir' must be specified."
    
    if dataset_dir: # Use dataset_dir
        dataset_dir = Path(dataset_dir)
        dataset_name = dataset_dir.name
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory '{dataset_dir}' does not exist.")
        if not (dataset_dir / 'label.csv').exists():
            raise FileNotFoundError(f"Label file '{dataset_dir}/label.csv' does not exist.")
        label_df = pd.read_csv(
            dataset_dir / 'label.csv',
            dtype={'clip_id': str, 'video_id': str, 'text': str}
        )
        return label_df, dataset_dir
    else: # Use dataset_name
        # check the dataset_root_dir
        # dataset_root_dir is required, it is the parent dir of dataset_name
        if dataset_root_dir is None:
            raise ValueError("Dataset root directory is not specified.")
        dataset_root_dir = Path(dataset_root_dir)
        if not dataset_root_dir.exists():
            raise FileNotFoundError(f"Dataset root directory '{dataset_root_dir}' does not exist.")
        
        try: # Try to locate label.csv according to global dataset config file
            with open(dataset_root_dir / 'config.json', 'r') as f:
                dataset_config_all = json.load(f)
            dataset_config = dataset_config_all[dataset_name]
            label_file = dataset_root_dir / dataset_config['label_path']

        except: # If failed, try to locate label.csv using joined path
            label_file = dataset_root_dir / dataset_name / 'label.csv'
        if not label_file.exists():
            raise FileNotFoundError(f"Label file '{label_file}' does not exist.")

        label_df = pd.read_csv(
            label_file,
            dtype={'clip_id': str, 'video_id': str, 'text': str}
        )
        return label_df, label_file.parent

def main(args):
    assert (args.dataset_name and args.dataset_root_dir) is not None or args.dataset_dir is not None, "Either 'dataset_name' or 'dataset_dir' must be specified."
    label_df, label_file_parent = read_label_file(args.dataset_name, args.dataset_root_dir, None)
    data={}
    for index, row in tqdm(label_df.iterrows(), total=label_df.shape[0], position=0, leave=True, desc=f"Processing {args.dataset_name}"):
        video_id, clip_id, text, label, label_T, label_A, label_V, annotation, split = \
            row['video_id'], row['clip_id'], row['text'], \
            row['label'], row['label_T'], row['label_A'], \
            row['label_V'], row['annotation'], row['mode']
        cur_id = video_id + '$_$' + clip_id
        video_path = label_file_parent / 'Raw' / video_id / (clip_id + '.mp4')
        assert video_path.exists(), f"Video file {video_path} does not exist"
        # extract keyframe
        keyframe_dir = label_file_parent / 'Raw_V' / video_id / (clip_id + '.jpg') 
        keyframe_dir.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_extract(video_path, str(keyframe_dir), mode='image')

        # extract speaker face
        face_dir = label_file_parent / 'Face' / video_id / clip_id
        face_dir.parent.mkdir(parents=True, exist_ok=True)
        save_faces(keyframe_dir, face_dir)

        # extract audio 
        audio_path = label_file_parent/ 'Raw_A' / video_id / (clip_id + '.wav')
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio = ffmpeg_extract(str(video_path), str(audio_path), mode='audio')

        if split not in data:
            data[split] = {}

        data[split][cur_id] = {
            "video_path": f"Raw/{video_id}/{clip_id}.mp4",
            "audio_path": f"Raw_A/{video_id}/{clip_id}.wav",
            "image_path": f"Raw_V/{video_id}/{clip_id}.jpg",
            "face_dir": f"Face/{video_id}/{clip_id}",
            "text": text,
            'sentiment': annotation,
            'regression_labels': label,
            't_labels': label_T,
            'a_labels': label_A,
            'v_labels': label_V
        }

    out_dir = f"{args.dataset_root_dir}/{args.dataset_name}/Processed/{args.dataset_name}_data.pkl"
    save_result(data, out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MMSA data")
    parser.add_argument("--dataset_root_dir", type=str, default="datasets/ERC", help="Root directory of datasets")
    parser.add_argument("--dataset_dir", type=str, default="datasets/ERC/CMU_MOSEI", help="Path to dataset folder")
    parser.add_argument("--dataset_name", type=str, default="CMU_MOSEI", help="Name of the dataset")
    args = parser.parse_args()
    main(args)
