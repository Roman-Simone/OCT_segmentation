#!/bin/bash
#SBATCH -n 1 
#SBATCH -c 1 
#SBATCH -N 1 
#SBATCH --output=output.txt
#SBATCH -t 10:00:00
#SBATCH -p edu-20h
#SBATCH --gres=gpu:1


conda init 
source ~/.bashrc
conda activate oct

# nnUNetv2_predict -i /home/simone.roman/OCT_segmentation/data/RETOUCH_dataset/RETOUCH_PROCESSED_NNUNET/nnUNet_raw/Dataset003_Total/imagesTe -o /home/simone.roman/OCT_segmentation/prediction_2d -d 003 -c 2d -f 4 --save_probabilities


# nnUNetv2_train 003 3d_lowres 4

# nnUNetv2_plan_and_preprocess -d 001 002 000 003 --verify_dataset_integrity

# 2d 3d_fullres 3d_cascade_fullres 3d_lowres 3d_cascade_lowres

# nnUNetv2_predict -i /home/simone.roman/OCT_segmentation/data/RETOUCH_dataset/RETOUCH_PROCESSED_NNUNET/nnUNet_raw/Dataset003_Total/imagesTe -o /home/simone.roman/OCT_segmentation/prediction_2d -d 003 -c 2d -f 4 --save_probabilities
nnUNetv2_predict -i /home/simone.roman/OCT_segmentation/data/RETOUCH_dataset/RETOUCH_PROCESSED_NNUNET/nnUNet_raw/Dataset003_Total/imagesTe -o /home/simone.roman/OCT_segmentation/prediction_3d_fullres -d 003 -c 3d_fullres -f 4 --save_probabilities
nnUNetv2_predict -i /home/simone.roman/OCT_segmentation/data/RETOUCH_dataset/RETOUCH_PROCESSED_NNUNET/nnUNet_raw/Dataset003_Total/imagesTe -o /home/simone.roman/OCT_segmentation/prediction_3d_lowres -d 003 -c 3d_lowres -f 4 --save_probabilities


###### MEDNEXT

# mednextv1_plan_and_preprocess -t 000 001 002 003 -pl3d ExperimentPlanner3D_v21_customTargetSpacing_1x1x1 -pl2d ExperimentPlanner2D_v21_customTargetSpacing_1x1x1



##### SAMED

python src\\models\\samEd\\train.py --root_path "C:\\Users\\simone.roman\\project\\OCT_segmentation\\data\\RETOUCH_dataset\\RETOUCH_PROCESSED_SAMED\\samed_npz\\Dataset003_Total" --output "C:\\Users\\simone.roman\\project\\OCT_segmentation\\data\\RETOUCH_dataset\\RETOUCH_PROCESSED_SAMED\\results" --dataset "Retouch" --num_classes 4 --n_gpu 1


