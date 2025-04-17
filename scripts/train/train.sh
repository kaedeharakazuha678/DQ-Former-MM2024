#!/bin/bash
clear
script_path="MoE-MERC"
cd "$script_path"
num_gpus=1
source "$script_path"/scripts/configs/config.sh ${num_gpus}

export WANDB_DISABLE=False

project="DQ_Former"
time=$(date "+%Y%m%d-%H%M%S")
model_name='dq_tav'
dataset_name='MELD'
class_type="emotion"
label_type="discrete"
modalities='tav'
text_model_name='bert_base_uncased'
audio_model_name='wavlm_base'
vision_model_name='clip_base'
fusion_layers=4
num_layers=4
context_window_size=0
bottleneck_len=16
num_heads=8
batch_size=8
gradient_accumulation_steps=1
tau=0.1
t_loss_weight=0.5
a_loss_weight=0.5
v_loss_weight=0.2
f_loss_weight=1

if [[ $text_model_name == *"base"* ]]; then
  intermediate_dim=768
fi

if [[ $text_model_name == *"large"* ]]; then
  intermediate_dim=1024
fi

if [[ $text_model_name == *"llama"* ]]; then
  intermediate_dim=4096
fi

text_variable="${text_model_name}"
text_model=$(eval echo "\$$text_variable")
echo "Path for $text_variable: $text_model"
audio_variable="${audio_model_name}"
audio_model=$(eval echo "\$$audio_variable")
echo "Path for $audio_variable: $audio_model"
vision_variable="${vision_model_name}"
vision_model=$(eval echo "\$$vision_variable")
echo "Path for $vision_variable: $vision_model"

group_name=${dataset_name}-${model_name}-${modalities}
run_name=${text_model_name}-${audio_model_name}-${vision_model_name}-"l"-${num_layers}-"b"-${bottleneck_len}-"c"-${context_window_size}-"tau"-${tau}-"t"-${t_loss_weight}-"a"-${a_loss_weight}-"v"-${v_loss_weight}-"f"-${f_loss_weight}
echo "Group name: $group_name"
echo "Run name: $run_name"


python train.py \
    --wandb_project ${project} \
    --wandb_run_name ${run_name} \
    --wandb_group_name ${group_name} \
    --model_name ${model_name} \
    --dataset_name ${dataset_name} \
    --dataset_root_dir ${DATASET_ROOT_DIR} \
    --label_type ${label_type} \
    --class_type ${class_type} \
    --vision_model ${vision_model} \
    --text_model ${text_model} \
    --audio_model ${audio_model} \
    --audio_processor ${wav2vec2_base} \
    --num_layers ${num_layers} \
    --num_heads ${num_heads} \
    --fusion_layers ${fusion_layers} \
    --context_window_size ${context_window_size} \
    --bottleneck_len ${bottleneck_len} \
    --intermediate_dim ${intermediate_dim} \
    --batch_size ${batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --num_epochs 20 \
    --learning_rate 1e-5 \
    --warmup_proportion 0.01 \
    --max_seq_len 512 \
    --time_durations 10.0 \
    --tau ${tau} \
    --early_stopping_patience 20 \
    --do_train True \
    --save_total_limit 2 \
    --t_loss_weight ${t_loss_weight} \
    --a_loss_weight ${a_loss_weight} \
    --v_loss_weight ${v_loss_weight} \
    --f_loss_weight ${f_loss_weight} \
    --enable_modality_embedding True \
    --modalities ${modalities} \
    --wandb_disable ${WANDB_DISABLE} \
