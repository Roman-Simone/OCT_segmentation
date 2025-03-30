import os
import h5py
import json
import shutil
import numpy as np
import nibabel as nib
from tqdm import tqdm
from glob import glob
from icecream import ic
import SimpleITK as sitk



TRAIN = 0.8
TEST = 0.2

class ProcessDataset:

    def __init__(self, input_data: str, output_data: str):
        self.input_dir = input_data
        self.output_dir = output_data
        self.hashmap = {
            0: 0,  # Background
            1: 1,  # Intraretinal fluid (IRF)
            2: 2,  # Subretinal fluid (SRF)
            3: 3   # Pigment Epithelial Detachment (PED)
        }

    def process_retouch_raw(self):
        """
            Process the RETOUCH dataset to convert it to the format required by nnU-Net.
        """
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "samed_raw"), exist_ok=True)
        samed_raw_dir = os.path.join(self.output_dir, "samed_raw")

        # Take the training directories
        dirs = self.take_retouch_dir()
        
        
        dataset_total_path = f"{samed_raw_dir}/Dataset00{len(dirs)}_Total"
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
            dataset_path = f"{samed_raw_dir}/Dataset00{pos}_{name_dir}"
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
            self.create_json_retouch(dataset_path, pos)
        self.create_json_retouch(dataset_total_path, len(dirs))
        
        

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

    def convert_mhd_to_nii(self, output_dir, mhd_file, patient_id, output_type, flagLabel=False):
        """
            Convert an MHD file to NIfTI format.
        """
        image = sitk.ReadImage(mhd_file)
        patient_id = str.replace(patient_id, "TRAIN", "")
        output_path = os.path.join(output_dir, output_type, f"img0{patient_id}.nii.gz")
        if flagLabel:
            output_path = os.path.join(output_dir, output_type, f"lbl0{patient_id}.nii.gz")
        sitk.WriteImage(image, output_path)

    def create_json_retouch(self, dataset_path, pos):
        """
            Create the JSON file for a specific dataset inside its directory.
        """
        json_file = os.path.join(dataset_path, 'dataset.json')

        dataset_info = {
            "channel_names": {
                "0": "OCT"  
            },
            "labels": {
                "background": 0,
                "Intraretinal fluid (IRF)": 1,
                "Subretinal fluid (SRF)": 2,
                "Pigment Epithelial Detachment (PED)": 3
            },
            "numTraining": self.get_num_training_cases(dataset_path),
            "file_ending": ".nii.gz",
            "overwrite_image_reader_writer": "SimpleITKIO"  # Optional, can be omitted if not needed
        }

        # Save the dataset_info as a JSON file
        with open(json_file, 'w') as f:
            json.dump(dataset_info, f, indent=4)

    def get_num_training_cases(self, dataset_path):
        """
            Get the number of training cases in the dataset.
        """
        training_images_dir = os.path.join(dataset_path, 'imagesTr')
        patient_dirs = os.listdir(training_images_dir)
        return len(patient_dirs)

    def preprocess_retouch_npz(self) -> None:

        #navigate the directory inside the output dir
        
        for dir in os.listdir(f"{self.output_dir}/samed_raw"):
        
            image_files = sorted(glob(f"{self.output_dir}/samed_raw/{dir}/imagesTr/*.nii.gz"))
            label_files = sorted(glob(f"{self.output_dir}/samed_raw/{dir}/labelsTr/*.nii.gz"))

            a_min, a_max = -125, 275
            b_min, b_max = 0.0, 1.0

            pbar = tqdm(zip(image_files, label_files), total=len(image_files),  desc=f"Processing npz {dir}", unit="patient")
            for image_file, label_file in pbar:
                # **/imgXXXX.nii.gz -> parse XXXX
                number = image_file.split('/')[-1][3:7]


                image_data = nib.load(image_file).get_fdata()
                label_data = nib.load(label_file).get_fdata()

                image_data = image_data.astype(np.float32)
                label_data = label_data.astype(np.float32)

                image_data = np.clip(image_data, a_min, a_max)
                # if args.use_normalize:
                #     assert a_max != a_min
                #     image_data = (image_data - a_min) / (a_max - a_min)

                H, W, D = image_data.shape

                image_data = np.transpose(image_data, (2, 1, 0))  # [D, W, H]
                label_data = np.transpose(label_data, (2, 1, 0))

                counter = 0
                for k in sorted(self.hashmap.keys()):
                    assert counter == k
                    counter += 1
                    label_data[label_data == k] = self.hashmap[k]

                for dep in range(D):
                    os.makedirs(f"{self.output_dir}/samed_npz/{dir}", exist_ok=True)
                    save_path = f"{self.output_dir}/samed_npz/{dir}/case{number}_slice{dep:03d}.npz"
                    np.savez(save_path, label=label_data[dep,:,:], image=image_data[dep,:,:])
            pbar.close()


if __name__ == "__main__":
    # Define the input and output directories
    input_data = "data/RETOUCH_dataset/RETOUCH"
    output_data = "data/RETOUCH_dataset/RETOUCH_PROCESSED_SAMED"

    # Create an instance of ProcessDataset and start the process
    processor = ProcessDataset(input_data, output_data)
    processor.process_retouch_raw()
    processor.preprocess_retouch_npz()
