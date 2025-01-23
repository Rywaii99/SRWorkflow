#!/bin/bash
#JSUB -n 2
#JSUB -q gpu
#JSUB -gpgpu 1
#JSUB -m "gpu05 gpu06 gpu07 gpu08 gpu15 gpu16 gpu20"
#JSUB -e output/error.%J
#JSUB -o output/output.%J
source /apps/software/anaconda3/etc/profile.d/conda.sh
conda activate py38tr1121cu116
unset PYTHONPATH
python train.py -opt=options/train/002_EDSR_x2_f256b32_AID_100k_B16G1P64.yml

