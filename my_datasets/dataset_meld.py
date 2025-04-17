import librosa
import cv2
import pickle
import copy
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers import AutoProcessor
from utils.constant_utils import *
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Union, Tuple

class MELDDataset(Dataset):
    def __init__(self, 
                 args, 
                 dataset_path, 
                 up_sample=False, 
                 meta_instruction: Optional[str]=None):
        self.args = args
        self.bottleneck_len = 16 if args.bottleneck_len is None else args.bottleneck_len
        self.time_durations = args.time_durations
        self.context_window_size = args.context_window_size
        self.class_type = args.class_type
        self.meta_path= os.path.join(args.dataset_root_dir, args.dataset_name)
        self.meta_instruction = meta_instruction
        self.tokenizer = AutoTokenizer.from_pretrained(args.text_model)
        self.sep_token = self.tokenizer.sep_token
        self.audio_processor = self.get_audio_processor()
        self.vision_processor = self.get_vision_processor()
    
        with open(dataset_path, 'rb') as file:
            raw_data = pickle.load(file)  
        
        if self.context_window_size > 0:  
            # self.samples = self.get_data_with_history(raw_data, self.context_window_size)
            self.samples = self._reformat_data_with_window_history(raw_data, self.context_window_size)
        else:
            self.samples = raw_data
        
        if up_sample:
            self.samples = self.upsampling(self.samples)

    def upsampling(self, data):
        # upsampling the minority class
        emotion_count = Counter([sample['emotion'] for sample in data.values()])
        max_count = max(emotion_count.values())
        for emotion in emotion_count.keys():
            emotion_count[emotion] = max_count - emotion_count[emotion]
        
        new_data = data.copy()
        for key, sample in list(data.items()):
            while emotion_count[sample['emotion']] > 0:
                new_data[str(key) + str(emotion_count[sample['emotion']])] = sample
                emotion_count[sample['emotion']] -= 1
        return new_data

    def get_audio_processor(self):
        if "wavlm" in self.args.audio_model or "hubert" in self.args.audio_model or 'wav2vec2' in self.args.audio_model:
            self.audio_processor = AutoProcessor.from_pretrained(self.args.audio_processor)
        else:
            self.audio_processor = AutoProcessor.from_pretrained(self.args.audio_model)
        return self.audio_processor  

    def get_vision_processor(self):
        self.vision_processor = AutoProcessor.from_pretrained(self.args.vision_model)
        return self.vision_processor

    def get_data_with_history(self, data, context_window_size=5):
        data = copy.deepcopy(data)
        reformatted_data = {}
        dialogue_histories = {}
        
        sorted_data = sorted(data.items(), key=lambda x: (int(x[1]['dialogue_id']), int(x[1]['utterance_id'])))
        
        for entry_name, sample in sorted_data:
            dialogue_id = sample['dialogue_id']
            if dialogue_id not in reformatted_data:
                reformatted_data[dialogue_id] = []
                dialogue_histories[dialogue_id] = []
            
            current_text = f"{sample['speaker']}:{sample['text']} "
            dialogue_histories[dialogue_id].append(current_text)
            
            # Create the history text before limiting the dialogue history size
            history_text = ' '.join(dialogue_histories[dialogue_id][:-1])
            
            # Limit the dialogue history to the context window size
            dialogue_histories[dialogue_id] = dialogue_histories[dialogue_id][-context_window_size:]  
            
            if history_text:
                sample['text'] = f"{history_text} ### {current_text}"
            else:
                sample['text'] = f"{current_text}"
        
            reformatted_data[dialogue_id].append(sample)

        # Flatten the dictionary to match the original structure
        flattened_data = {}
        for dialogue_id, samples in reformatted_data.items():
            for idx, sample in enumerate(samples):
                entry_name = f"dia{dialogue_id}_utt{idx}"
                flattened_data[entry_name] = sample

        return flattened_data

    def _reformat_data_with_window_history(self, data, context_window_size=5):
        data = copy.deepcopy(data)
        reformatted_data = {}
        dialogue_histories = {}
        
        sorted_data = sorted(data.items(), key=lambda x: (int(x[1]['dialogue_id']), int(x[1]['utterance_id'])))
        
        for entry_name, sample in sorted_data:
            dialogue_id = sample['dialogue_id']
            if dialogue_id not in reformatted_data:
                reformatted_data[dialogue_id] = []
                dialogue_histories[dialogue_id] = []
            
            current_text = f"{sample['speaker']}:{sample['text']} "
            dialogue_histories[dialogue_id].append(current_text)
            
            # Create the history text before limiting the dialogue history size
            history_text = ' '.join(dialogue_histories[dialogue_id][:-1])
            
            # Limit the dialogue history to the context window size
            dialogue_histories[dialogue_id] = dialogue_histories[dialogue_id][-context_window_size:]  
            
            if history_text:
                sample['text'] = f" Dialogue history   {history_text} ### the target utterance : {current_text}"
            else:
                sample['text'] = f"{current_text}"
        
            reformatted_data[dialogue_id].append(sample)

        # Flatten the dictionary to match the original structure
        flattened_data = {}
        for dialogue_id, samples in reformatted_data.items():
            for idx, sample in enumerate(samples):
                entry_name = f"dia{dialogue_id}_utt{idx}"
                flattened_data[entry_name] = sample

        return flattened_data
             
    def pad_or_truncate_waveform(self, waveform, sample_rate, target_duration=5.0):
        """
        Pads or truncates a waveform to a given duration.

        Args:
            - waveform (np.array): The input waveform.
            - sample_rate (int): The sample rate of the waveform (e.g., 44100 for 44.1 kHz).
            - target_duration (float): The target duration in seconds. Default is 5.0 seconds.

        Returns:
            - np.array: The padded or truncated waveform.
        """
        
        # Calculate the number of samples for the target duration.
        target_samples = int(target_duration * sample_rate)

        # If waveform is shorter than the target duration, pad it.
        if len(waveform) < target_samples:
            padded_waveform = np.zeros(target_samples)
            padded_waveform[:len(waveform)] = waveform
            return padded_waveform
        # If waveform is longer than the target duration, truncate it.
        elif len(waveform) > target_samples:
            return waveform[:target_samples]
        # If waveform is exactly the target duration, return it as is.
        else:
            return waveform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry_name, sample = list(self.samples.items())[idx]
        emotion_mapping = MELD_EMOTION_MAPPING
        sentiment_mapping = SENTIMENT_MAPPING
        return_data = {}

        text = sample['text'].lower().replace("###", self.tokenizer.sep_token)
        encoded_text = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.args.max_seq_len, return_tensors="pt", add_special_tokens=True)           
        return_data["input_ids"] = encoded_text["input_ids"].squeeze(0)
        return_data["attention_mask"] = encoded_text["attention_mask"].squeeze(0)
        
        audio_path=os.path.join(self.meta_path, sample['audio_path'])
        waveform, sample_rate = librosa.load(audio_path, sr=16000)     
        audio_inputs = self.audio_processor(waveform, return_tensors="pt", sampling_rate=sample_rate, return_attention_mask=True)
        return_data["audio_value"] = audio_inputs.input_features.squeeze() if 'whisper' in self.args.audio_model else audio_inputs.input_values.squeeze()
        return_data["audio_attention_mask"] = audio_inputs.attention_mask.squeeze()
        return_data["wav_path"] = sample['audio_path']

        image_path=os.path.join(self.meta_path, sample['image_path'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        vision_inputs = self.vision_processor(images=image, return_tensors="pt", return_attention_mask=True)
        return_data['pixel_values'] = vision_inputs.pixel_values.squeeze()
        return_data["image_path"] = sample['image_path']
        
        return_data["text"] = text
        return_data["emotion"] = emotion_mapping[sample['emotion']]
        return_data["sentiment"] = sentiment_mapping[sample['sentiment']]
        return_data["speaker"] = sample['speaker']
        return_data["utterance_id"] = sample['utterance_id']
        return_data["dialogue_id"] = sample['dialogue_id']
        return_data["labels"] = emotion_mapping[sample['emotion']] if self.class_type == 'emotion' else sentiment_mapping[sample['sentiment']]
        return_data["queries"] = [i for i in range(self.bottleneck_len)]
        return return_data
    
