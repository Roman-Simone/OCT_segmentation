@echo off
:: === ENVIRONMENT VARIABLES FOR MedNeXt ===
set "nnUNet_raw_data_base=C:\Users\simone.roman\project\OCT_segmentation\data\RETOUCH_dataset\RETOUCH_PROCESSED_MEDNEXT"
set "nnUNet_preprocessed=C:\Users\simone.roman\project\OCT_segmentation\data\RETOUCH_dataset\RETOUCH_PROCESSED_MEDNEXT\nnUNet_preprocessed"
set "RESULTS_FOLDER=C:\Users\simone.roman\project\OCT_segmentation\data\RETOUCH_dataset\RETOUCH_PROCESSED_MEDNEXT"
set "nnUNet_results=C:\Users\simone.roman\project\OCT_segmentation\data\RETOUCH_dataset\RETOUCH_PROCESSED_NNUNET\nnUNet_results"
set "nnUNet_raw=C:\Users\simone.roman\project\OCT_segmentation\data\RETOUCH_dataset\RETOUCH_PROCESSED_NNUNET\nnUNet_raw"

:: === ACTIVATE CONDA ENVIRONMENT ===
call conda activate oct

:: === RUN TRAINING ===
:: Training with MedNeXt S model with 3x3x3 kernel
:: Replace 'Task003_RETOUCH' with your task number if different
:: Replace '0' with your fold number if different

mednextv1_train 2d nnUNetTrainerV2_MedNeXt_S_kernel3 Task003_Total 4 -p nnUNetPlansv2.1_trgSp_1x1x1 

echo Training completed!
pause