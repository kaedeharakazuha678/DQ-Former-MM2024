#!/bin/bash
cd /home/jye/MERC/MoE-MERC/scripts
export CUDA_VISIBLE_DEVICES='2'
bash eval/infer.sh seed_35_0414_1/checkpoint-1400 
sleep 10
bash eval/infer.sh seed_35_0414_1/checkpoint-3220 
sleep 10
bash eval/infer.sh seed_42_0413_1/checkpoint-1600 
sleep 10
bash eval/infer.sh seed_42_0413_1/checkpoint-3220 
sleep 10

# bash eval/infer.sh IEMOCAP 0 seed_42_0409_1/checkpoint-3800 
# bash eval/infer.sh IEMOCAP 0 seed_42_0409_1/checkpoint-5800

# bash eval/infer.sh IEMOCAP 1 seed_42_0409_1/checkpoint-3200 
# bash eval/infer.sh IEMOCAP 1 seed_42_0409_1/checkpoint-5200

# bash eval/infer.sh IEMOCAP 2 seed_42_0409_1/checkpoint-3600 
# bash eval/infer.sh IEMOCAP 2 seed_42_0409_1/checkpoint-5600

# bash eval/infer.sh IEMOCAP 3 seed_42_0409_1/checkpoint-3200 
# bash eval/infer.sh IEMOCAP 3 seed_42_0409_1/checkpoint-5200

# bash eval/infer.sh IEMOCAP 4 seed_42_0409_1/checkpoint-5400 
# bash eval/infer.sh IEMOCAP 4 seed_42_0409_1/checkpoint-7400

# bash eval/infer.sh IEMOCAP 6 seed_42_0410_1/checkpoint-4700 
# bash eval/infer.sh IEMOCAP 6 seed_42_0410_1/checkpoint-6700

# bash eval/infer.sh IEMOCAP 7 seed_42_0410_1/checkpoint-3200 
# bash eval/infer.sh IEMOCAP 7 seed_42_0410_1/checkpoint-5200

# bash eval/infer.sh IEMOCAP 8 seed_42_0410_1/checkpoint-5600 
# bash eval/infer.sh IEMOCAP 8 seed_42_0410_1/checkpoint-7600


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