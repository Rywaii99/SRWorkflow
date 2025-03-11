#!/bin/bash
#JSUB -n 2
#JSUB -q gpu
#JSUB -gpgpu 1
#JSUB -m "gpu09 gpu10 gpu13 gpu14 gpu21 gpu22 gpu25 gpu26 gpu03 gpu05 gpu06 gpu07 gpu15 gpu20"
#JSUB -e output/error.%J
#JSUB -o output/output.%J
source /apps/software/anaconda3/etc/profile.d/conda.sh
conda activate py38tr1121cu116
unset PYTHONPATH
python train.py -opt=options/train/007_HSENet_x3_AID_100k_B16G1P96.yml

