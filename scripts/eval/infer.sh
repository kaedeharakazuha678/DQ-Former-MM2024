#!/bin/bash
clear
script_path="MoE-MERC"
cd "$script_path"
num_gpus=1
source "$script_path"/scripts/configs/config.sh ${num_gpus}

model_name='dq_tav'
dataset_name='MELD'
class_type="emotion"
label_type="discrete"
text_model_name='bert_base_uncased'
audio_model_name='wavlm_base'
vision_model_name='clip_base'
num_layers=4
fusion_layers=4
checkpoint_name=$1
context_window_size=0
modalities='tav'
bottleneck_len=16
num_heads=8
cv_fold=5
tau=0.1
t_loss_weight=0.5
a_loss_weight=0.5
v_loss_weight=0.2
f_loss_weight=1


if [[ $text_model_name == *"base"* ]]; then
  intermediate_dim=768
else
  intermediate_dim=1024
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
eval_model_path="outputs/checkpoints/${dataset_name}_${class_type}_${label_type}/${model_name}-${modalities}/${run_name}/${checkpoint_name}/pytorch_model.bin"

python test.py \
    --wandb_run_name ${run_name} \
    --dataset_name ${dataset_name} \
    --dataset_root_dir ${DATASET_ROOT_DIR} \
    --label_type ${label_type} \
    --class_type ${class_type} \
    --text_model ${text_model} \
    --audio_model ${audio_model} \
    --vision_model ${vision_model} \
    --audio_processor ${wav2vec2_base} \
    --fusion_layers ${fusion_layers} \
    --num_layers ${num_layers} \
    --context_window_size ${context_window_size} \
    --bottleneck_len ${bottleneck_len} \
    --intermediate_dim ${intermediate_dim} \
    --batch_size 1 \
    --max_seq_len 512 \
    --time_durations 15.0 \
    --eval_model_path ${eval_model_path} \
    --model_name ${model_name} \
    --do_save False \
    --need_weights False \
    --num_heads ${num_heads} \
    --cv_fold ${cv_fold} \
    --t_loss_weight ${t_loss_weight} \
    --a_loss_weight ${a_loss_weight} \
    --f_loss_weight ${f_loss_weight} \
    --tau ${tau} \
    --modalities ${modalities} \
