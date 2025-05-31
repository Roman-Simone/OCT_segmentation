@echo off
:: === VARIABILI AMBIENTE PER medNeXt ===
set "nnUNet_raw_data_base=C:\Users\simone.roman\project\OCT_segmentation\data\RETOUCH_dataset\RETOUCH_PROCESSED_MEDNEXT"
set "nnUNet_preprocessed=C:\Users\simone.roman\project\OCT_segmentation\data\RETOUCH_dataset\RETOUCH_PROCESSED_MEDNEXT\nnUNet_preprocessed"
set "RESULTS_FOLDER=C:\Users\simone.roman\project\OCT_segmentation\data\RETOUCH_dataset\RETOUCH_PROCESSED_MEDNEXT"
set "nnUNet_results=C:\Users\simone.roman\project\OCT_segmentation\data\RETOUCH_dataset\RETOUCH_PROCESSED_NNUNET\nnUNet_results"
set "nnUNet_raw=C:\Users\simone.roman\project\OCT_segmentation\data\RETOUCH_dataset\RETOUCH_PROCESSED_NNUNET\nnUNet_raw"
:: === ATTIVA L'AMBIENTE CONDA ===
call conda activate oct

:: === ESECUZIONE SCRIPT PYTHON ===
python data\RETOUCH_dataset\process_RETOUCH_mednext.py

:: === LANCIA PREPROCESSING MEDNEXT ===

:: === LANCIA PREPROCESSING MEDNEXT ===
:retry
mednextv1_plan_and_preprocess -t 000 001 002 003 -pl3d ExperimentPlanner3D_v21_customTargetSpacing_1x1x1 -pl2d ExperimentPlanner2D_v21_customTargetSpacing_1x1x1
if %ERRORLEVEL% neq 0 (
    echo Errore durante l'esecuzione di mednextv1_plan_and_preprocess. Eseguo fix_medNet_plan.
    python data\RETOUCH_dataset\process_RETOUCH_mednext.py --fix
    goto retry
)
python data\RETOUCH_dataset\process_RETOUCH_mednext.py --clean
echo Completato con successo.