#region import and initialize
import argparse
import os
import sys
import time
import torch
from tqdm import tqdm, trange
from tqdm.contrib import tenumerate
from torch.utils.data import DataLoader
import torch.nn.functional as F
from my_datasets import *
from my_models import *
from utils.constant_utils import *
from utils.args_utils import my_args
import pandas as pd
import pickle
import numpy as np
import librosa
from utils.metrics_utils import MyComputeMetrics
from transformers import EvalPrediction, AutoTokenizer, AutoProcessor
from loguru import logger
sys.path.append(os.pardir) 
#endregion

class EvaluationMetrics(MyComputeMetrics):
    def __init__(self, args):
        super(EvaluationMetrics, self).__init__(args)
    
    def forward(self, all_labels, all_preds):
        p = EvalPrediction(predictions=np.array(all_preds), label_ids=np.array(all_labels))
        eval_stats=self.get_eval_stat(p) 
        eval_results = self.transfer_results_to_df(eval_stats)
        return eval_results
    
        
class FeatureSaver:
    def __init__(self, args) -> None:
        self.args=args
        self.feature_dict = {}
    
    def get_model_features(self, output_dict):
        for key, value in output_dict.items():
            if key.endswith('feature') or key.endswith('confs'):
                if torch.is_tensor(value):
                    self.feature_dict.setdefault(key, []).append(value.cpu().numpy())
                else:
                    self.feature_dict.setdefault(key, []).append(value)
    
    def get_attention_weights(self, output_dict):
        if 'attn_weights' in output_dict.keys():
            if torch.is_tensor(output_dict['attn_weights']):
                self.feature_dict.setdefault('attn_weights', []).append(output_dict['attn_weights'].cpu().numpy())
            else:
                self.feature_dict.setdefault('attn_weights', []).append(output_dict['attn_weights'])
            
    def get_metadata(self, batch):
        for key, value in batch.items():
            if key in ['text', 'audio_path', 'labels', 'wav_path', 'image_path']:
                self.feature_dict.setdefault(key, []).append(value)
    
    def get_prediction(self, message, name):
        self.feature_dict.setdefault(name, []).append(message)
      
    
    def save_features(self):
        feature_path = os.path.join(
            self.args.output_dir,
            f"features/{self.args.dataset_name}/{self.args.model_name}/{self.args.wandb_run_name}"
        )
        os.makedirs(feature_path, exist_ok=True)
        path = os.path.join(feature_path, f"{self.args.formatted_time}.pkl")
        # save features to pickle file
        with open(path, 'wb') as f:
            pickle.dump(self.feature_dict, f)
        print(f"save features to {path}")
        

def load_model_and_dataset(args):
    print(">>> Loading dataset...")
    _, _, test_dataset = load_dataset(args)
    data_collator = DataCollator(args)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator)

    print(">>> Loading model...")
    model = load_model(args)
    if args.eval_model_path != '':
        logger.info(f">>> Loading model from {args.eval_model_path}")
        model.load_state_dict(torch.load(args.eval_model_path))
    else:
        print("!!!Warning: No model path is provided, model will be initialized randomly.")
    model = model.to('cuda')
    model.eval()
    return model, test_dataloader


def easy_test(args):
    model, test_dataloader = load_model_and_dataset(args)

    print(">>> Start Tesing...")
    all_preds, all_labels = [], []

    with torch.no_grad():
        for step, batch in tenumerate(test_dataloader, total=len(test_dataloader), desc="Testing"):
            texts, input_ids, attention_mask, labels, audio_features, queries, pixel_values, audio_attention_mask = (
                batch['text'],
                batch['input_ids'].to('cuda'),
                batch['attention_mask'].to('cuda'), 
                batch['labels'].to('cuda'), 
                batch['audio_value'].to('cuda'),
                batch['queries'].to('cuda'),
                batch['pixel_values'].to('cuda'),
                batch['audio_attention_mask'].to('cuda')
            )

            output = model(audio_value=audio_features, 
                           input_ids=input_ids, 
                           attention_mask=attention_mask, 
                           queries=queries, 
                           text=texts, 
                           labels=labels,
                           pixel_values=pixel_values,
                           audio_attention_mask=audio_attention_mask)
            
            logits = output['logits']
            all_preds.extend(logits.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        evaluation = EvaluationMetrics(args)
        result_table = evaluation.forward(all_labels, all_preds)
        logger.info(result_table.to_markdown()) 
 
    return all_labels, all_preds, result_table

def test_multimodal(args):
    model,test_dataloader = load_model_and_dataset(args)
    
    print(">>> Testing and evaluating...")
    all_preds, all_labels = [], []
    all_a_preds, all_t_preds, all_v_preds = [], [], []
    all_texts, all_audio_paths, all_image_paths = [], [], []
    all_emotions = []
    
    model_saver = FeatureSaver(args)
    with torch.no_grad():
        for step, batch in tenumerate(test_dataloader, total=len(test_dataloader), desc="Testing"):
            texts, input_ids, attention_mask, audio_paths, audio_values, pixel_values, image_paths, queries, labels = (
                batch['text'], batch['input_ids'].to('cuda'), batch['attention_mask'].to('cuda'),
                batch['wav_path'], batch['audio_value'].to('cuda'), 
                batch['pixel_values'].to('cuda'), batch['image_path'],
                batch['queries'].to('cuda'), batch['labels'].to('cuda')
            )

            output = model(audio_value=audio_values, 
                           input_ids=input_ids, 
                           attention_mask=attention_mask, 
                           queries=queries, 
                           text=texts, 
                           labels=labels, 
                           pixel_values=pixel_values)
            
            logits = output['logits']
            preds = torch.argmax(logits, dim=-1)
            a_logits = output['return_dict']['a_logits']
            a_preds = torch.argmax(a_logits, dim=-1)
            t_logits = output['return_dict']['t_logits']
            t_preds = torch.argmax(t_logits, dim=-1)
            v_logits = output['return_dict']['v_logits']
            v_preds = torch.argmax(v_logits, dim=-1)

            a_confs = output['return_dict']['a_confs']
            t_confs = output['return_dict']['t_confs']
            v_confs = output['return_dict']['v_confs']

            all_texts.extend(texts)
            all_audio_paths.extend(audio_paths)
            all_image_paths.extend(image_paths)
            all_preds.extend(logits.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_a_preds.extend(a_logits.cpu().numpy())
            all_t_preds.extend(t_logits.cpu().numpy())
            all_v_preds.extend(v_logits.cpu().numpy())
    
            #region save model features
            model_saver.get_model_features(output['return_dict'])
            model_saver.get_metadata(batch)
            model_saver.get_attention_weights(output['return_dict'])
            model_saver.get_prediction(preds.cpu().numpy(), 'preds')
            model_saver.get_prediction(a_preds.cpu().numpy(), 'a_preds')
            model_saver.get_prediction(t_preds.cpu().numpy(), 't_preds')
            model_saver.get_prediction(v_preds.cpu().numpy(), 'v_preds')
            #endregion

            for text, pred, a_pred, t_pred, v_pred, label, logit, a_logit, t_logit, v_logit, audio_path in zip(texts, preds, a_preds, t_preds, v_preds, labels, logits, a_logits, t_logits, v_logits, audio_paths):
                logger.info('='*100)
                logger.info(f"Text: {text}")
                logger.info(f"Audio: {audio_path}")
                logger.info(f"Pred: {pred}")
                logger.info(f"Audio Pred: {a_pred}")
                logger.info(f"Text Pred: {t_pred}")
                logger.info(f"Vision Pred: {v_pred}")
                logger.info(f"Label: {label}")
                logger.info(f"Logits: {[round(num, 4) for num in logit.tolist()]}")
                logger.info(f"Confidence: {round(torch.max(logit).item(), 4)}")
                logger.info(f"Audio Logits: {[round(num, 4) for num in a_logit.tolist()]}")
                logger.info(f"Audio Confidence: {round(torch.max(a_logit).item(), 4)}")
                logger.info(f"Text Logits: {[round(num, 4) for num in t_logit.tolist()]}")
                logger.info(f"Text Confidence: {round(torch.max(t_logit).item(), 4)}")
                logger.info(f"Vision Logits: {[round(num, 4) for num in v_logit.tolist()]}")
                logger.info(f"Vision Confidence: {round(torch.max(v_logit).item(), 4)}")
                     
        if args.do_save:
            model_saver.save_features()

        #region get evaluation metrics
        evaluation = EvaluationMetrics(args)
        f_results = evaluation.forward(all_labels, all_preds)
        a_results = evaluation.forward(all_labels, all_a_preds)
        t_results = evaluation.forward(all_labels, all_t_preds)
        v_results = evaluation.forward(all_labels, all_v_preds)
        results = pd.concat([f_results, a_results, t_results, v_results], ignore_index=True).drop_duplicates()
        logger.info(results.to_markdown())
        #endregion

    return all_preds, all_a_preds, all_t_preds, all_v_preds, all_labels, all_texts, all_audio_paths, all_image_paths, results


def main(args):
    #region save test log 
    local_time = time.localtime(time.time())
    args.formatted_time = time.strftime("%Y%m%d-%H%M%S", local_time)
    result_save_path = f'{args.output_dir}/test_logs/{args.dataset_name}_{args.class_type}_{args.label_type}/{args.model_name}/{args.wandb_run_name}/{args.formatted_time}'
    os.makedirs(result_save_path, exist_ok=True)
    logger.add(f'{args.output_dir}/test_logs/test.log', format="{time} {level} {message}", level="INFO")
    logger.info(args)
    #endregion
    
    # all_labels, all_preds, result_table = easy_test(args)
    test_multimodal(args)


def single_infer(args):
    #region process raw_data
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    audio_processor = AutoProcessor.from_pretrained(args.audio_processor) if "wisper" not in args.audio_model else AutoProcessor.from_pretrained(args.audio_model)
    wave, sample_rate = librosa.load(audio_path, sr=16000)
    encoded_audio = audio_processor(wave, return_tensors="pt", sampling_rate=sample_rate)
    audio_input = encoded_audio.input_features if 'whisper' in args.audio_model else encoded_audio.input_values

    encoded_text = tokenizer(raw_text, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded_text.input_ids
    attention_mask = encoded_text.attention_mask
    queries = [i for i in range(args.bottleneck_len)]
    #endregion

    #region load model 
    model = load_model(args)
    if args.eval_model_path != '':
        logger.info(f">>> Loading model from {args.eval_model_path}")
        model.load_state_dict(torch.load(args.eval_model_path))
    else:
        print("!!!Warning: No model path is provided, model will be initialized randomly.")
    model = model.to('cuda')
    model.eval()
    #endregion

    #region get results
    output = model(audio_input, input_ids, attention_mask, queries=queries, text=raw_text)
    logits = output['logits']
    print(logits)
    #endregion


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MM-ERC  settings")

    parser.add_argument("--wandb_run_name", type=str, default='MMERC', help="Run name")
    parser.add_argument("--eval_model_path", type=str, default='')
    parser.add_argument("--batch", default=True, type=bool, help="for single sample infer or for batch test")
    #region testing setting
    parser.add_argument("--do_save", default=False, help="whether to save the model features for visualization t-sne")
    parser.add_argument("--do_train", default=False, help="set this to False for testing")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--plot_save_path", type=str, default='../outputs/plot', help="Path to save plot")
    #endregion
    args = my_args(parser).parse_args()
    if args.batch:
        main(args)
    else:
        single_infer(args)