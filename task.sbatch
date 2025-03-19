#!/bin/bash
#SBATCH -n 1 
#SBATCH -c 1 
#SBATCH -N 1 
#SBATCH --output=output3.txt
#SBATCH -t 10:00:00
#SBATCH -p edu-20h
#SBATCH --gres=gpu:1


conda init 
source ~/.bashrc
conda activate oct


# nnUNetv2_train 003 3d_fullres 4

nnUNetv2_predict -i /home/simone.roman/OCT_segmentation/data/RETOUCH_PROCESSED_NNUNET/nnUNet_raw/Dataset002_Set/imagesTr -o /home/simone.roman/OCT_segmentation/output -d 003 -c 3d_fullres -f 4 --save_probabilities