from .dataset_iemocap_4_class import IEMOCAP_4_Class
from .dataset_iemocap_6_class import IEMOCAP_6_Class
from .dataset_meld import MELDDataset
from .dataset_mmsa import MMSADataset
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

#NOTE replace with our data path

def load_dataset(args):
    dataset_dir = Path(args.dataset_root_dir) / args.dataset_name / "Processed"

    if args.dataset_name == 'MELD':
        train_dataset = MELDDataset(args, str(dataset_dir / "train_data_33.pkl"))
        dev_dataset = MELDDataset(args, str(dataset_dir / "dev_data_33.pkl"))
        test_dataset = MELDDataset(args, str(dataset_dir / "test_data_33.pkl"))

    elif args.dataset_name == 'IEMOCAP':
        if args.class_type == '4_ways':
            train_dataset = IEMOCAP_4_Class(args, str(dataset_dir / "processed_data.csv"), split='train', cv_fold=args.cv_fold)
            dev_dataset = IEMOCAP_4_Class(args, str(dataset_dir / "processed_data.csv"), split='val', cv_fold=args.cv_fold)    
            test_dataset = IEMOCAP_4_Class(args, str(dataset_dir / "processed_data.csv"), split='test', cv_fold=args.cv_fold)
        else:
            train_dataset = IEMOCAP_6_Class(args, str(dataset_dir / "processed_data.csv"), split='train', cv_fold=args.cv_fold)
            dev_dataset = IEMOCAP_6_Class(args, str(dataset_dir / "processed_data.csv"), split='val', cv_fold=args.cv_fold)    
            test_dataset = IEMOCAP_6_Class(args, str(dataset_dir / "processed_data.csv"), split='test', cv_fold=args.cv_fold)

    elif 'CMU' in args.dataset_name:
        train_dataset = MMSADataset(args, str(dataset_dir / f"{args.dataset_name}_data.pkl"), up_sample=False, split='train')
        dev_dataset = MMSADataset(args, str(dataset_dir / f"{args.dataset_name}_data.pkl"), up_sample=False, split='valid')
        test_dataset = MMSADataset(args, str(dataset_dir / f"{args.dataset_name}_data.pkl"), up_sample=False, split='test')

    elif 'CH_SIMS' in args.dataset_name:
        train_dataset = MMSADataset(args, str(dataset_dir / f"{args.dataset_name}_data.pkl"), up_sample=False, split='train')
        dev_dataset = MMSADataset(args, str(dataset_dir / f"{args.dataset_name}_data.pkl"), up_sample=False, split='valid')
        test_dataset = MMSADataset(args, str(dataset_dir / f"{args.dataset_name}_data.pkl"), up_sample=False, split='test')

    else:
        raise ValueError(f"Invalid dataset name {args.dataset_name}")
    
    return train_dataset, dev_dataset, test_dataset

class DataCollator:
    def __init__(self, args):
        self.args = args

    def __call__(self, batch):
        keys = batch[0].keys()
        batched = {}

        for key in keys:
            if 'audio' in key:
                if 'whisper' in self.args.audio_model:
                    batched[key] = torch.stack([sample[key] for sample in batch])
                else:
                    global_max_length = int(16000 * self.args.time_durations)
                    global_min_length = 16000
                    batch_max_length = min(global_max_length, max(min(global_max_length, sample[key].size(0)) for sample in batch))
                    batch_max_length = max(batch_max_length, global_min_length)
                    batched[key] = torch.stack([F.pad(sample[key][:batch_max_length], pad=(0, batch_max_length - min(sample[key].size(0), batch_max_length))) for sample in batch])

            elif isinstance(batch[0][key], torch.Tensor):
                batched[key] = torch.stack([sample[key] for sample in batch])
            elif isinstance(batch[0][key], np.ndarray):
                batched[key] = torch.tensor(np.stack([sample[key] for sample in batch]))
            elif isinstance(batch[0][key], (int, float, list)):
                batched[key] = torch.tensor([sample[key] for sample in batch])
            else:
                batched[key] = [sample[key] for sample in batch]

        return batched