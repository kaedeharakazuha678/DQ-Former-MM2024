import copy
import os
import librosa
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoTokenizer, LlamaTokenizer, BertTokenizer
from utils.constant_utils import *
from sklearn.model_selection import train_test_split
import json
from collections import Counter

_SAMPLE_RATE = 16000

class IEMOCAP_4_Class(Dataset):
    def __init__(self, args, dataset_path, pass_unknown=True, split='train', cv_fold=0, up_sampling=True):
        self.args = args
        self.bottleneck_len = args.bottleneck_len
        self.label_type = args.label_type
        self.base_path = os.path.join(args.dataset_root_dir, args.dataset_name)  # path to IEOCAP_full_release
        self.max_length = args.max_seq_len
        self.context_window_size = args.context_window_size
        self.split_type = args.split_type
        self.pass_unknown = pass_unknown 
        self.cv_fold = cv_fold
        self.tokenizer = AutoTokenizer.from_pretrained(args.text_model)
        self.sep_token = self.tokenizer.sep_token
        self.audio_processor = self.get_audio_processor()
        self.data = self.read_csv_data(dataset_path)
        if self.context_window_size > 0:
            self.data = self.get_data_with_history(self.context_window_size)
            # self.data = self._reformat_data_with_window_history(self.context_window_size)
        else:
            self.data = self.data

        if self.pass_unknown:
            self.data = self.remove_unknown_label()

        if self.split_type == 'random':
            self.train_ids, self.val_ids, self.test_ids = self.split_dataset_ramdomly()
            if split in ['train', 'val', 'test']:
                self.data = [sample for sample in self.data if sample['utterance_id'] in getattr(self, f'{split}_ids')]
        else:
            self.data = self.split_dataset(split)

        if split == 'train' and up_sampling:
            self.data = self.upsampling(self.data)

        self.data = self.filter_dataset(self.data, ['fru', 'hap'])

    def upsampling(self, data):
        # upsampling the minority class
        emotion_count = Counter([sample['emotion'] for sample in data])
        max_count = max(emotion_count.values())
        for emotion in emotion_count.keys():
            emotion_count[emotion] = max_count - emotion_count[emotion]
        for sample in data:
            if emotion_count[sample['emotion']] > 0:
                data.append(sample)
                emotion_count[sample['emotion']] -= 1
        return data

    def filter_dataset(self, data, emotion):
        # delete the samples with the emotion in [frustration, happy]
        return [sample for sample in data if sample['emotion'] not in emotion]

    def read_csv_data(self, csv_path):
        df = pd.read_csv(csv_path)
        data = []
        for i in range(len(df)):
            sample = {
                'audio_path': os.path.join(self.base_path, df.at[i, 'audio_path']),
                'text': df.at[i, 'text'],
                'emotion': df.at[i, 'emotion'],
                'dialog_id': df.at[i, 'dialog_id'],
                'utterance_id': df.at[i, 'utterance_id'],
                'speaker': df.at[i, 'speaker'],
                'vad': df.at[i, 'vad']
            }
            data.append(sample)
        return data

    def get_data_with_history(self, context_window_size=5):
        reformatted_data = []
        dialogue_histories = {}
        data = copy.deepcopy(self.data)
        for sample in data:
            dialogue_id = sample['dialog_id']
            if dialogue_id not in dialogue_histories:
                dialogue_histories[dialogue_id] = []

            current_text = f"{sample['speaker']}: {sample['text']} "
            dialogue_histories[dialogue_id].append(current_text)

            # Create the history text 
            history_text = ' '.join(dialogue_histories[dialogue_id][:-1])

            # Limit the dialogue history to the context window size
            dialogue_histories[dialogue_id] = dialogue_histories[dialogue_id][-context_window_size:]

            if history_text:
                sample['text'] = f"{history_text} ### {current_text}"
            else:
                sample['text'] = f"{current_text}"

            reformatted_data.append(sample)

        return reformatted_data

    def _reformat_data_with_window_history(self, context_window_size=5):
        reformatted_data = []
        dialogue_histories = {}
        data = copy.deepcopy(self.data)
        for sample in data:
            dialogue_id = sample['dialog_id']
            if dialogue_id not in dialogue_histories:
                dialogue_histories[dialogue_id] = []

            current_text = f"{sample['speaker']}: {sample['text']} "
            dialogue_histories[dialogue_id].append(current_text)

            # Create the history text 
            history_text = ' '.join(dialogue_histories[dialogue_id][:-1])

            # Limit the dialogue history to the context window size
            dialogue_histories[dialogue_id] = dialogue_histories[dialogue_id][-context_window_size:]

            if history_text:
                sample['text'] = f" Dialogue history  {history_text} ### the target utterance {current_text}"
            else:
                sample['text'] = f" Dialogue history is none ### the target utterance {current_text}"

            reformatted_data.append(sample)

        return reformatted_data

    def remove_unknown_label(self):
        df = pd.DataFrame(self.data)
        # Filter out rows with 'unknown' emotion
        df = df[df['emotion'] != 'unknown']
        return df.to_dict(orient='records')  

    def get_audio_processor(self):
        if "wavlm" in self.args.audio_model or "hubert" in self.args.audio_model or "wav2vec2" in self.args.audio_model:
            return AutoProcessor.from_pretrained(self.args.audio_processor)
        else:
            return AutoProcessor.from_pretrained(self.args.audio_model)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return_dict = {}
        emotion_mapping = IEMOCAP_EMOTION_MAPPING_4_CLASS
        sample = self.data[index]
        audio_path = sample['audio_path']
        text = sample['text'].lower().replace('###', self.sep_token)

        wavframe, sample_rate = librosa.load(audio_path, sr=_SAMPLE_RATE)
        audio_inputs = self.audio_processor(wavframe, sampling_rate=sample_rate, return_tensors="pt", return_attention_mask=True) 
        return_dict["audio_value"] = audio_inputs.input_features.squeeze(0) if 'whisper' in self.args.audio_model else audio_inputs.input_values.squeeze()
        return_dict["audio_attention_mask"] = audio_inputs.attention_mask.squeeze()
        return_dict['wav_path'] = audio_path

        encoded_text = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.args.max_seq_len, return_tensors="pt", add_special_tokens=True)
        return_dict['input_ids'] = encoded_text['input_ids'].squeeze(0)
        return_dict['attention_mask'] = encoded_text['attention_mask'].squeeze(0)
        return_dict['text'] = text

        return_dict['emotion'] = emotion_mapping[sample['emotion']]
        return_dict['vad'] = json.loads(sample['vad'])
        return_dict['labels'] = emotion_mapping[sample['emotion']] if self.label_type == 'discrete' else json.loads(sample['vad'])
        return_dict["queries"] = [i for i in range(self.bottleneck_len)]
        return return_dict

    def split_dataset_according_to_dialog_id(self, test_size=0.1, validation_size=0.1, random_state=42):
        dialog_ids = list(set([sample['dialog_id'] for sample in self.data]))

        train_dialog_ids, test_dialog_ids = train_test_split(dialog_ids, test_size=test_size, random_state=random_state)
        train_dialog_ids, val_dialog_ids = train_test_split(train_dialog_ids, test_size=validation_size, random_state=random_state)

        return train_dialog_ids, val_dialog_ids, test_dialog_ids
    
    def split_dataset_according_to_session(self, cv_fold=0):
        dialog_ids = list(set([sample['dialog_id'] for sample in self.data]))
        train_dialog_ids, test_dialog_ids = [], []
        for dialog_id in dialog_ids:
            if dialog_id[:5] == f'Ses0{cv_fold}':
                test_dialog_ids.append(dialog_id)
            else:
                train_dialog_ids.append(dialog_id)
        val_dialog_ids = test_dialog_ids
        return train_dialog_ids, val_dialog_ids, test_dialog_ids

    def split_dataset_ramdomly(self, test_size=0.1, validation_size=0.1, random_state=42):
        utterance_ids = list([sample['utterance_id'] for sample in self.data])
        train_utterance_ids, test_utterance_ids = train_test_split(utterance_ids, test_size=test_size, random_state=random_state)
        train_utterance_ids, val_utterance_ids = train_test_split(train_utterance_ids, test_size=validation_size, random_state=random_state)
        return train_utterance_ids, val_utterance_ids, test_utterance_ids

    def split_dataset(self, split):
        if self.split_type == 'dialog':
            split_message = ">>> Using dialog split, split the dataset according to dialog id."
            self.train_dialog_ids, self.val_dialog_ids, self.test_dialog_ids = self.get_fold_split(self.cv_fold)

        elif self.split_type == 'session':
            split_message = ">>> Using session split, split the dataset according to session id."
            self.train_dialog_ids, self.val_dialog_ids, self.test_dialog_ids = self.split_dataset_according_to_session(self.cv_fold)

        else:
            raise ValueError(f">>> Unsupported split type: {self.split_type}")

        print(split_message)
        if split in ['train', 'val', 'test']:
            self.data = [sample for sample in self.data if sample['dialog_id'] in getattr(self, f'{split}_dialog_ids')]
        else:
            raise ValueError(f"Unsupported split value: {split}")
        return self.data

    def get_fold_split(self, cv_fold=0):
        # accroding to dialog id
        fold_split_file = 'split.json'
        with open(fold_split_file, 'r') as f:
            split = json.load(f)
        
        train_dialog_ids = split[str(cv_fold)]['train_dialog_ids']
        val_dialog_ids = split[str(cv_fold)]['val_dialog_ids']
        test_dialog_ids = split[str(cv_fold)]['test_dialog_ids']

        return train_dialog_ids, val_dialog_ids, test_dialog_ids






