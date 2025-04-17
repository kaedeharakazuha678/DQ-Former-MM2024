import pickle
import matplotlib.pyplot as plt
import copy
import os
import pandas as pd
from tqdm import tqdm
from transformers import LlamaTokenizer, LlamaModel, LlamaForCausalLM
from utils.constant_utils import MELD_EMOTION_MAPPING
from sklearn.metrics import classification_report
from my_datasets import load_dataset

def get_llama_model(model_path):
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path, device_map='auto')
    return tokenizer, model

def read_data(dataset_path):
    with open(dataset_path, 'rb') as file:
        data = pickle.load(file)
    return data
    
def _reformat_data_with_window_history(data, context_window_size=5):
    reformatted_data = {}
    dialogue_histories = {}
    data = copy.deepcopy(data)
    sorted_data = sorted(data.items(), key=lambda x: (int(x[1]['dialogue_id']), int(x[1]['utterance_id'])))
    
    meta_instruction = "### According to the conversation history, you are excepted to generate the answer of the target utterance and recognize the emotion of it. The emotion label can be selected from <netural, joy, sadness, anger, fear, disgust, surprise>. \n### Conversation history \n"
    
    for entry_name, sample in sorted_data:
        dialogue_id = sample['dialogue_id']
        if dialogue_id not in reformatted_data:
            reformatted_data[dialogue_id] = []
            dialogue_histories[dialogue_id] = []
        
        current_text = f"{sample['speaker']}:{sample['text']}\n"
        dialogue_histories[dialogue_id].append(current_text)
        
        # Create the history text before limiting the dialogue history size
        history_text = ' '.join(dialogue_histories[dialogue_id][:-1])
        
        # Limit the dialogue history to the context window size
        dialogue_histories[dialogue_id] = dialogue_histories[dialogue_id][-context_window_size:]  
        
        if history_text:
            sample['text'] = f"{meta_instruction} {history_text}### The target utterence \n {current_text}### The emotion of the target utterance is \n"
        else:
            # sample['text'] = current_text
            sample['text'] = f"{meta_instruction} {None}### The target utterence \n {current_text}### The emotion of the target utterance is \n"
        
        reformatted_data[dialogue_id].append(sample)

    # Flatten the dictionary to match the original structure
    flattened_data = {}
    for dialogue_id, samples in reformatted_data.items():
        for idx, sample in enumerate(samples):
            entry_name = f"dia{dialogue_id}_utt{idx}"
            flattened_data[entry_name] = sample

    return flattened_data


def evaluate(generated_answers, labels):
    emotion_mapping = MELD_EMOTION_MAPPING
    generated_answer = [emotion_mapping[answer.lower()] for answer in generated_answer]
    labels = [emotion_mapping[answer.lower()] for answer in labels]
    report = classification_report(labels, generated_answers, output_dict=True)
    pd = pd.DataFrame(report).transpose()
    return pd
    

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
model_path = "models/Llama-2-7b-chat-hf/"
dataset_path = "preprocessed_data/test_data.pkl"

tokenizer, model = get_llama_model(model_path)

raw_data = read_data(dataset_path)
data_new = _reformat_data_with_window_history(raw_data)
test_texts = [sample['text'] for sample in data_new.values()]
labels = [sample['emotion'] for sample in data_new.values()]

generated_answers = []
Processbar = tqdm(total=len(test_texts[:100]))
for test_text in test_texts:
    inputs = tokenizer(test_text, return_tensors='pt')
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)
    output=model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=100, top_k=50, top_p=0.95, do_sample=True, num_return_sequences=1)
    generated_answer = tokenizer.decode(output[0][-2])
    generated_answers.append(generated_answer)
    print(generated_answer)
    Processbar.update(1)
    
report = evaluate(generated_answers, labels[:100])
print(report)