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

class MMSADataset(Dataset):
    def __init__(self, args, dataset_path, up_sample=True, split='train'):
        self.args = args
        self.bottleneck_len = args.bottleneck_len
        self.time_durations = args.time_durations
        self.context_window_size = args.context_window_size
        self.class_type = args.class_type
        self.meta_path= os.path.join(args.dataset_root_dir, args.dataset_name)
        self.tokenizer = AutoTokenizer.from_pretrained(args.text_model)
        self.sep_token = self.tokenizer.sep_token
        self.audio_processor = self.get_audio_processor()
        self.vision_processor = self.get_vision_processor()
    
        with open(dataset_path, 'rb') as file:
            raw_data = pickle.load(file)  
        raw_data = raw_data[split]
        
        if self.context_window_size > 0:  
            self.samples = self.reformatted_data(raw_data, self.context_window_size)
            # self.samples = self._reformatted_data_with_window_history(raw_data, self.context_window_size)
        else:
            self.samples = raw_data
        
        if up_sample:
            self.samples = self.upsampling(self.samples)

    def upsampling(self, data):
        # upsampling the minority class
        emotion_count = Counter([sample['sentiment'] for sample in data.values()])
        max_count = max(emotion_count.values())
        for emotion in emotion_count.keys():
            emotion_count[emotion] = max_count - emotion_count[emotion]
        
        new_data = data.copy()
        for key, sample in list(data.items()):
            while emotion_count[sample['sentiment']] > 0:
                new_data[str(key) + str(emotion_count[sample['sentiment']])] = sample
                emotion_count[sample['sentiment']] -= 1
        return new_data

    def reformatted_data(self, data, context_window_size=5):
        data = copy.deepcopy(data)
        reformatted_data = {}
        context_histories = {}
        for entry_name, sample in data.items():
            video_id, clip_id = entry_name.split('$_$')
            if video_id not in context_histories:
                context_histories[video_id] = []
            context_histories[video_id].append((int(clip_id), sample))
        
        for video_id, context in context_histories.items():
            context.sort(key=lambda x: x[0])  # Sort by clip_id
            for i, (clip_id, sample) in enumerate(context):
                context_window = context[max(0, i - context_window_size): i+1]
                context_text = " ".join([c_sample[1]['text'] for c_sample in context_window[:-1]])
                context_text = f"{context_text} ### {context_window[-1][1]['text']}"

                reformatted_data[f"{video_id}$_${clip_id}"] = {
                    'video_path': f'Raw/{video_id}/{clip_id}.mp4',
                    'audio_path': f'Raw_A/{video_id}/{clip_id}.wav',
                    'image_path': f'Raw_V/{video_id}/{clip_id}.jpg',
                    'face_dir': f'Face/{video_id}/{clip_id}',
                    'text': context_text,
                    'sentiment': sample['sentiment'],
                    'regression_labels': sample['regression_labels'],
                    't_labels': sample['t_labels'],
                    'a_labels': sample['a_labels'],
                    'v_labels': sample['v_labels'],
                }
        
        return reformatted_data

    def _reformatted_data_with_window_history(self, data, context_window_size=5):
        data = copy.deepcopy(data)
        reformatted_data = {}
        context_histories = {}
        for entry_name, sample in data.items():
            video_id, clip_id = entry_name.split('$_$')
            if video_id not in context_histories:
                context_histories[video_id] = []
            context_histories[video_id].append((int(clip_id), sample))
        
        for video_id, context in context_histories.items():
            context.sort(key=lambda x: x[0])  # Sort by clip_id
            for i, (clip_id, sample) in enumerate(context):
                context_window = context[max(0, i - context_window_size): i+1]
                context_text = " ".join([c_sample[1]['text'] for c_sample in context_window[:-1]])
                context_text = f"Dialogue history is : {context_text} ### The target utterance is {context_window[-1][1]['text']}"

                reformatted_data[f"{video_id}$_${clip_id}"] = {
                    'video_path': f'Raw/{video_id}/{clip_id}.mp4',
                    'audio_path': f'Raw_A/{video_id}/{clip_id}.wav',

                    'text': context_text,
                    'sentiment': sample['sentiment'],
                    'regression_labels': sample['regression_labels'],
                }
        
        return reformatted_data

    def get_audio_processor(self):
        if "wavlm" in self.args.audio_model or "hubert" in self.args.audio_model or 'wav2vec2' in self.args.audio_model:
            self.audio_processor = AutoProcessor.from_pretrained(self.args.audio_processor)
        else:
            self.audio_processor = AutoProcessor.from_pretrained(self.args.audio_model)
        return self.audio_processor  

    def get_vision_processor(self):
        self.vision_processor = AutoProcessor.from_pretrained(self.args.vision_model)
        return self.vision_processor    
             
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
        # import pdb; pdb.set_trace()
        sentiment_mapping = SENTIMENT_MAPPING
        video_id, clip_id = entry_name.split('$_$')
        return_data = {}

        text = sample['text'].lower().replace('###', self.sep_token)
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
        return_data["sentiment"] = sample['sentiment']
        return_data["video_id"] = video_id
        return_data["clip_id"] = clip_id
        return_data["labels"] = sentiment_mapping[sample['sentiment'].lower()] if self.args.label_type == 'discrete' else sample['regression_labels']
        return_data["queries"] = [i for i in range(self.bottleneck_len)]
        return return_data

