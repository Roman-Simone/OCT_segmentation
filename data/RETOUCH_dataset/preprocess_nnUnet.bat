@echo off
set "BASE_PATH=C:\Users\simone.roman\project\OCT_SEGMENTATION"
set "RAW_PATH=%BASE_PATH%\data\RETOUCH_dataset\RETOUCH_PROCESSED_NNUNET\nnUNet_raw"
set "PREPROCESSED_PATH=%BASE_PATH%\data\RETOUCH_dataset\RETOUCH_PROCESSED_NNUNET\nnUNet_preprocessed"
set "RESULTS_PATH=%BASE_PATH%\data\RETOUCH_dataset\RETOUCH_PROCESSED_NNUNET\nnUNet_results"

:: Imposta le variabili per la sessione attuale
set "nnUNet_raw=%RAW_PATH%"
set "nnUNet_preprocessed=%PREPROCESSED_PATH%"
set "nnUNet_results=%RESULTS_PATH%"

nnUNetv2_plan_and_preprocess -d 000 001 002 003 --verify_dataset_integrity
