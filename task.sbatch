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


nnUNetv2_train 003 3d_fullres 4