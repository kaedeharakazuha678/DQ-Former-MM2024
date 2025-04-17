#!/bin/bash
cd /home/jye/MERC/MoE-MERC/scripts
# export CUDA_VISIBLE_DEVICES='4'

# bash eval/infer_1.sh single_stream
# sleep 10
# bash eval/infer_1.sh sa_mask_text
# sleep 10
# bash eval/infer_1.sh sa_mask_vision
# sleep 10
# bash eval/infer_1.sh sa_mask_audio
# sleep 10
# bash eval/infer_1.sh sa_mask_text
# sleep 10
# bash eval/infer_1.sh sa_mask_vision
# sleep 10
bash train/train_2.sh MELD
sleep 10
bash train/train_2.sh IEMOCAP





# bash eval/infer.sh dq_tav IEMOCAP 1 seed_42_0409_1/checkpoint-3200 tav
# bash eval/infer.sh dq_tav IEMOCAP 1 seed_42_0409_1/checkpoint-5200 tav

# bash eval/infer.sh dq_tav IEMOCAP 2 seed_42_0409_1/checkpoint-3600 tav
# bash eval/infer.sh dq_tav IEMOCAP 2 seed_42_0409_1/checkpoint-5600 tav

# bash eval/infer.sh dq_tav IEMOCAP 3 seed_42_0409_1/checkpoint-3200 tav
# bash eval/infer.sh dq_tav IEMOCAP 3 seed_42_0409_1/checkpoint-5200 tav


# bash eval/infer.sh dq_tav IEMOCAP 4 seed_42_0409_1/checkpoint-5400 tav
# bash eval/infer.sh dq_tav IEMOCAP 4 seed_42_0409_1/checkpoint-7400 tav

# bash eval/infer.sh dq_tav IEMOCAP 6 seed_42_0410_1/checkpoint-4700 tav
# bash eval/infer.sh dq_tav IEMOCAP 6 seed_42_0410_1/checkpoint-6700 tav

# bash eval/infer.sh dq_tav IEMOCAP 7 seed_42_0410_1/checkpoint-3200 tav
# bash eval/infer.sh dq_tav IEMOCAP 7 seed_42_0410_1/checkpoint-5200 tav

# bash eval/infer.sh dq_tav IEMOCAP 8 seed_42_0410_1/checkpoint-5600 tav
# bash eval/infer.sh dq_tav IEMOCAP 8 seed_42_0410_1/checkpoint-7600 tav


# bash eval/infer.sh dq_tav MELD 0 seed_42_0410_1/checkpoint-3400 tav
# bash eval/infer.sh dq_tav MELD 0 seed_42_0410_1/checkpoint-5400 tav


# bash eval/infer.sh dq_tav MELD 1 seed_42_0410_1/checkpoint-3400 tav
# bash eval/infer.sh dq_tav MELD 1 seed_42_0410_1/checkpoint-5400 tav

# bash eval/infer.sh dq_tav MELD 2 seed_42_0410_1/checkpoint-3400 tav
# bash eval/infer.sh dq_tav MELD 2 seed_42_0410_1/checkpoint-5400 tav


# bash eval/infer.sh dq_tav MELD 3 seed_42_0410_1/checkpoint-2800 tav
# bash eval/infer.sh dq_tav MELD 3 seed_42_0410_1/checkpoint-4800 tav

# bash eval/infer.sh dq_tav MELD 4 seed_42_0410_1/checkpoint-3100 tav
# bash eval/infer.sh dq_tav MELD 4 seed_42_0410_1/checkpoint-5100 tav

# bash eval/infer.sh dq_tav MELD 6 seed_42_0410_1/checkpoint-3400 tav
# bash eval/infer.sh dq_tav MELD 6 seed_42_0410_1/checkpoint-5400 tav