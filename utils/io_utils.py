import os
import sys
import yaml
import pickle
import json
import time
from pathlib import Path

def save_as_pkl(data, filename):
    print(f"Saving data to {filename}")
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    print(f"Complete!")


def read_yaml(yaml_file):
    print(f"Reading data from {yaml_file}")
    with open(yaml_file, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data


def read_pkl(pkl_file):
    print(f"Reading data from {pkl_file}")
    with open(pkl_file, 'rb') as file:
        data = pickle.load(file)
    return data


def save_result(result, out_file):
    # demo: out_file = Path('path/output.pkl')
    out_file = Path(out_file)
    if out_file.exists():
        out_file_alt = out_file.parent / (out_file.stem + '_' + str(int(time.time())) + '.pkl')
        print(f"Output file '{out_file}' already exists. Saving to '{out_file_alt}' instead.")
        out_file = out_file_alt
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'wb') as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Feature file saved: {out_file}")