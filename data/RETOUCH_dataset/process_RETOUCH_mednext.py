import os
import json
import shutil
import argparse
from tqdm import tqdm
import SimpleITK as sitk



TRAIN = 0.8
TEST = 0.2

class ProcessDataset:

    def __init__(self, input_data: str, output_data: str):
        self.input_dir = input_data
        self.output_dir = output_data

    def processRetouch(self):
        """
            Process the RETOUCH dataset to convert it to the format required by nnU-Net.
        """
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "nnUNet_preprocessed"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "nnUNet_results"), exist_ok=True)
        unet_raw_dir = os.path.join(self.output_dir, "nnUNet_raw_data")

        # Take the training directories
        dirs = self.take_retouch_dir()
        print(len(dirs))
        
        
        dataset_total_path = f"{unet_raw_dir}/Task00{len(dirs)}_Total"
        os.makedirs(dataset_total_path, exist_ok=True)    
        os.makedirs(os.path.join(dataset_total_path, "imagesTr"), exist_ok=True)
        os.makedirs(os.path.join(dataset_total_path, "labelsTr"), exist_ok=True)
        os.makedirs(os.path.join(dataset_total_path, "imagesTe"), exist_ok=True)
        os.makedirs(os.path.join(dataset_total_path, "labelsTe"), exist_ok=True)

        for pos, dir in enumerate(dirs):
            
            patient_dirs = sorted(os.listdir(os.path.join(self.input_dir, dir)))

            name_dir = str.replace(dir, "Training", "")
            name_dir = str.replace(name_dir, "Retouch", "")
            name_dir = str.replace(name_dir, " ", "")
            dataset_path = f"{unet_raw_dir}/Task00{pos}_{name_dir}"
            os.makedirs(dataset_path, exist_ok=True)
            os.makedirs(os.path.join(dataset_path, "imagesTr"), exist_ok=True)
            os.makedirs(os.path.join(dataset_path, "labelsTr"), exist_ok=True)
            os.makedirs(os.path.join(dataset_path, "imagesTe"), exist_ok=True)
            os.makedirs(os.path.join(dataset_path, "labelsTe"), exist_ok=True)

            pos = 0
            for patient_dir in tqdm(patient_dirs, desc=f"Processing {dir}", unit="patient"):
                
                
                
                patient_path = os.path.join(self.input_dir, dir, patient_dir)

                oct_file = os.path.join(patient_path, 'oct.mhd')
                reference_file = os.path.join(patient_path, 'reference.mhd')

                if os.path.exists(oct_file) and os.path.exists(reference_file):
                    
                    # Check if the patient is in the training or testing set
                    if pos < TRAIN * len(patient_dirs):
                        pos += 1
                        train_test_dir = "Tr"
                    else:
                        train_test_dir = "Te"
                    # Convert the MHD files to NIfTI
                    self.convert_mhd_to_nii(dataset_path, oct_file, patient_dir, f"images{train_test_dir}")
                    self.convert_mhd_to_nii(dataset_path, reference_file, patient_dir, f"labels{train_test_dir}", flagLabel=True)
                    
                    self.convert_mhd_to_nii(dataset_total_path, oct_file, patient_dir, f"images{train_test_dir}")
                    self.convert_mhd_to_nii(dataset_total_path, reference_file, patient_dir, f"labels{train_test_dir}", flagLabel=True)
                    
                    

            # After processing, create the dataset JSON for this specific dataset
            self.create_json_retouch(dataset_path)
        self.create_json_retouch(dataset_total_path)

    def take_retouch_dir(self) -> list:
        """
            Take the directories that contain the training data of the RETOUCH dataset.
        """
        directories = []

        for root, dirs, _ in os.walk(self.input_dir):
            for dir_name in dirs:
                if "Training" in dir_name:
                    directories.append(dir_name)
        return directories

    def take_directories(self, path: str, key_word="") -> list:
        """
            Take the directories that contain the training data of the RETOUCH dataset.
        """
        directories = []

        for root, dirs, _ in os.walk(path):
            for dir_name in dirs:
                
                if key_word=="":
                    directories.append(dir_name)
                elif key_word in dir_name:
                    directories.append(dir_name)
                
        return directories
    
    
    def convert_mhd_to_nii(self, output_dir, mhd_file, patient_id, output_type, flagLabel=False):
        """
            Convert an MHD file to NIfTI format.
        """
        image = sitk.ReadImage(mhd_file)
        patient_id = str.replace(patient_id, "TRAIN", "")
        output_path = os.path.join(output_dir, output_type, f"TRAIN_{patient_id}_0000.nii.gz")
        if flagLabel:
            output_path = os.path.join(output_dir, output_type, f"TRAIN_{patient_id}.nii.gz")
        sitk.WriteImage(image, output_path)

    def create_json_retouch(self, dataset_path):
        training_files = sorted(os.listdir(os.path.join(dataset_path, 'imagesTr')))
        test_files = sorted(os.listdir(os.path.join(dataset_path, 'imagesTe')))
        
        dataset_info = {
            "name": "RETOUCH",
            "description": "oct and reference images",
            "reference": "",
            "licence": "",
            "modality": { "0": "OCT" },
            "labels": {
                "0":"background",
                "1":"Intraretinal fluid (IRF)",
                "2":"Subretinal fluid (SRF)",
                "3":"Pigment Epithelial Detachment (PED)"
            },
            "numTraining": len(training_files),
            "numTest": len(test_files),
            "file_ending": ".nii.gz",
            "overwrite_image_reader_writer": "SimpleITKIO",  # Optional, can be omitted if not needed
            "training": [
                {"image": f"./imagesTr/{img.replace('_0000', '')}", "label": f"./labelsTr/{img.replace('_0000', '')}"}
                for img in training_files if img.endswith(".nii.gz")
            ],
            "test": [f"./imagesTs/{img.replace('_0000', '')}" for img in test_files if img.endswith(".nii.gz")]
        }
        
        json_file = os.path.join(dataset_path, 'dataset.json')
        with open(json_file, 'w') as f:
            json.dump(dataset_info, f, indent=4)

    def get_num_training_cases(self, dataset_path):
        """
            Get the number of training cases in the dataset.
        """
        training_images_dir = os.path.join(dataset_path, 'imagesTr')
        patient_dirs = os.listdir(training_images_dir)
        return len(patient_dirs)



    def fix_medNet_plan(self):
        
        path = os.path.join(self.output_dir, "nnUNet_raw_data")
        
        dir_datasets = self.take_directories(path)  
        
        for dir_dataset in dir_datasets:
            
            dataset_path = os.path.join(path, dir_dataset)
            
            source_folder = os.path.join(dataset_path, "imagesTr")
            destination_folder = os.path.join(self.output_dir, "nnUNet_cropped_data")
            destination_folder = os.path.join(destination_folder, dir_dataset)


            # Elenco tutti i file nella cartella di origine
            for filename in os.listdir(source_folder):
                # Controlla se il file è un .npz o un .pkl
                if filename.endswith('.npz') or filename.endswith('.pkl'):
                    # Percorso completo dei file
                    source_file = os.path.join(source_folder, filename)
                    destination_file = os.path.join(destination_folder, filename)

                    try:
                        # Sposta il file
                        shutil.copy(source_file, destination_file)
                        print(f"File {filename} spostato con successo.")
                    except Exception as e:
                        print(f"Errore nel spostare il file {filename}: {e}")

    def clean_dir(self):
        
        path = os.path.join(self.output_dir, "nnUNet_raw_data")
        
        dir_datasets = self.take_directories(path)  
        
        for dir_dataset in dir_datasets:
            
            dataset_path = os.path.join(path, dir_dataset)
            
            source_folder = os.path.join(dataset_path, "imagesTr")

            # Elenco tutti i file nella cartella di origine
            for filename in os.listdir(source_folder):
                # Controlla se il file è un .npz o un .pkl
                if filename.endswith('.npz') or filename.endswith('.pkl'):
                    # Percorso completo dei file
                    source_file = os.path.join(source_folder, filename)
                    
                    try:
                        # Sposta il file
                        os.remove(source_file)
                        print(f"File {filename} spostato con successo.")
                    except Exception as e:
                        print(f"Errore nel spostare il file {filename}: {e}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Process RETOUCH dataset")
    parser.add_argument(
        "--fix", 
        action="store_true", 
        help="Run only the fix_medNet_plan function"
    )
    parser.add_argument(
        "--clean", 
        action="store_true", 
        help="Run only the cleaning function (e.g., delete specific files or directories)"
    )
    args = parser.parse_args()

    # Define the input and output directories
    input_data = "data/RETOUCH_dataset/RETOUCH"
    output_data = "data/RETOUCH_dataset/RETOUCH_PROCESSED_MEDNEXT"

    # Create an instance of ProcessDataset
    processor = ProcessDataset(input_data, output_data)

    if args.fix:
        processor.fix_medNet_plan()  # Run only the fix_medNet_plan function
    elif args.clean:
        processor.clean_dir()
    else:
        processor.processRetouch()  # Run the full processing
