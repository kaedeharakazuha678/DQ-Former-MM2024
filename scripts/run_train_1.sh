#!/bin/bash
cd /home/jye/MERC/MoE-MERC/scripts
export CUDA_VISIBLE_DEVICES="2"
bash train/train.sh 1 5 MELD
bash train/train.sh 2 5 MELD
bash train/train.sh 3 5 MELD
bash train/train.sh 4 5 MELD
bash train/train.sh 5 5 MELD
bash train/train.sh 6 5 MELD
bash train/train.sh 7 5 MELD
bash train/train.sh 8 5 MELD
bash train/train.sh 9 5 MELD
bash train/train.sh 10 5 MELD
bash train/train.sh 11 5 MELD
bash train/train.sh 12 5 MELD

bash train/train.sh 4 0 MELD
bash train/train.sh 4 1 MELD
bash train/train.sh 4 2 MELD
bash train/train.sh 4 3 MELD
bash train/train.sh 4 4 MELD
bash train/train.sh 4 5 MELD
bash train/train.sh 4 6 MELD
bash train/train.sh 4 7 MELD
bash train/train.sh 4 8 MELD

bash train/train.sh 2 5 IEMOCAP 