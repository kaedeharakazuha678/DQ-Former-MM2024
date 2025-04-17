#!/bin/bash

#region select gpus
num_gpus=$1  # Set the desired number of GPUs here

# wait for gpus

while true; do
    selected_gpus=$(python - <<END
import GPUtil

def get_free_gpus(num_gpus):
    try:
        available_gpus = GPUtil.getAvailable(order="memory", limit=num_gpus, maxLoad=0.1, maxMemory=0.1)
        if len(available_gpus) >= num_gpus:
            return available_gpus[:num_gpus]
        else:
            return None
    except Exception as e:
        print("Error while GPU selection:", str(e))
        return None

selected_gpus = get_free_gpus($num_gpus)  # Select the specified number of GPUs
if selected_gpus is not None:
    print(','.join(map(str, selected_gpus)))
else:
    print("Insufficient available GPUs.")
END
    )

    if [ ! -z "$selected_gpus" ] && [ "$selected_gpus" != "Insufficient available GPUs." ]; then
        export CUDA_VISIBLE_DEVICES=$selected_gpus
        echo "Setting GPU number to: $num_gpus"
        echo "Selected GPUs: $selected_gpus"
        break  # Break the loop if GPUs are available
    else
        echo "No available GPUs. Waiting for 1 minute..."
        sleep 60  # Wait for 1 minute before checking again
    fi
done

# export CUDA_VISIBLE_DEVICES='2'
#endregion

export WANDB_API_KEY=""

#region pretrain_model paths
# audio_encoders
whisper_small="{mask}/pretrained_model/openai/whisper-small"
hubert_base_ls960="{mask}/pretrained_model/facebook/hubert-base-ls960"
hubert_large_1160k="{mask}/pretrained_model/facebook/hubert-large-ll60k"
wav2vec2_base="{mask}/pretrained_model/facebook/wav2vec2-base"
wav2vec2_base_960h= "{mask}/pretrained_model/facebook/wav2vec2-base-960h"
wav2vec2_large="{mask}/pretrained_model/facebook/wav2vec2-large"
wavlm_base="{mask}/pretrained_model/microsoft/wavlm-base"
wavlm_base_plus="{mask}/pretrained_model/microsoft/wavlm-base-plus"
wavlm_large="{mask}/pretrained_model/microsoft/wavlm-large"
encodec_24khz="{mask}/pretrained_model/facebook/encodec_24khz"
# text_encoders
bert_large_uncased="{mask}/pretrained_model/bert-large-uncased"
bert_base_uncased="{mask}/pretrained_model/bert-base-uncased"
roberta_base="{mask}/pretrained_model/roberta-base"
roberta_large="{mask}/pretrained_model/roberta-large"
deberta_base="{mask}/pretrained_model/microsoft/deberta-base"
deberta_large="{mask}/pretrained_model/microsoft/deberta-v3-large"
llama2="{mask}/pretrained_model/Llama-2-7b-hf"
llama2_chat="{mask}/pretrained_model/Llama-2-7b-chat-hf"
llama3_8b="{mask}/pretrained_model/Meta-Llama-3-8B"
llama3_8b_chat="{mask}/pretrained_model/Meta-Llama-3-8B-Instruct"
# vision_encoders
vit_base_patch16_224="{mask}/pretrained_model/google/vit-base-patch16-224"
clip_base="{mask}/pretrained_model/openai/clip-vit-base-patch32"
clip_large="{mask}/pretrained_model/openai/clip-vit-large-patch14"
#endregion

DATASET_ROOT_DIR="{mask}/datasets/ERC"