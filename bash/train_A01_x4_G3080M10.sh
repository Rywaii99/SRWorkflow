#!/bin/bash
#JSUB -n 2
#JSUB -q gpu
#JSUB -gpgpu 1
#JSUB -m "gpu03"
#JSUB -e output/error.%J
#JSUB -o output/output.%J
source /apps/software/anaconda3/etc/profile.d/conda.sh
conda activate py38tr1121cu116
unset PYTHONPATH
python train.py -opt=options/train/A01_GCABNet_x4_w32enc16m12dec8_AID_100k_B16G1P128.yml

