import os
import re
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
        """
            Preprocess the NIfTI images to .npz slices for training.
            Applies clipping, normalization, axis transpose, and label remapping.
        """
        a_min, a_max = -125, 275
        b_min, b_max = 0.0, 1.0  # for normalization

        for dir in os.listdir(f"{self.output_dir}/samed_raw"):
            image_files = sorted(glob(f"{self.output_dir}/samed_raw/{dir}/imagesTr/*.nii.gz"))
            label_files = sorted(glob(f"{self.output_dir}/samed_raw/{dir}/labelsTr/*.nii.gz"))

            pbar = tqdm(zip(image_files, label_files), total=len(image_files),  desc=f"Processing npz {dir}", unit="patient")
            for image_file, label_file in pbar:
                number = image_file.split('/')[-1][3:7]

                image_data = nib.load(image_file).get_fdata().astype(np.float32)
                label_data = nib.load(label_file).get_fdata().astype(np.float32)

                # 📉 Clipping
                image_data = np.clip(image_data, a_min, a_max)

                # ⚖️ Normalizzazione in [0, 1]
                image_data = (image_data - a_min) / (a_max - a_min)

                # 🔁 Trasformazione assi: [H, W, D] -> [D, H, W]
                image_data = np.transpose(image_data, (2, 1, 0))
                label_data = np.transpose(label_data, (2, 1, 0))

                # 🧠 Rimappatura delle label (se necessario)
                for k, v in self.hashmap.items():
                    label_data[label_data == k] = v

                # 💾 Salvataggio slice-by-slice in .npz
                for dep in range(image_data.shape[0]):
                    os.makedirs(f"{self.output_dir}/samed_npz/{dir}", exist_ok=True)
                    save_path = f"{self.output_dir}/samed_npz/{dir}/case{number}_slice{dep:03d}.npz"
                    np.savez(save_path, label=label_data[dep, :, :], image=image_data[dep, :, :])
            pbar.close()

            
    def createlists_retouch(self) -> None:
        """
            Create the lists for the RETOUCH dataset.
        """
        if os.path.exists(f"{self.output_dir}/samed_lists"):
            shutil.rmtree(f"{self.output_dir}/samed_lists")
        os.makedirs(f"{self.output_dir}/samed_lists", exist_ok=True)

        # Read all the npz files in the samed_npz directory
        npz_files = sorted(glob(f"{self.output_dir}/samed_npz/Dataset003_Total/*.npz"))
        
        # take only the name of the file without the path and extension
        npz_files = [os.path.basename(file).replace('.npz', '') for file in npz_files]
        
        
        # Save the list of npz files in a text file
        with open(f"{self.output_dir}/samed_lists/train.txt", 'w') as f:
            for npz_file in npz_files:
                f.write(f"{npz_file}\n")
                
    def prepare_testset_h5(self):
        a_min, a_max = -125, 275
        output_dir = os.path.join(self.output_dir, 'samed_test_h5')
        os.makedirs(output_dir, exist_ok=True)

        for dataset_dir in os.listdir(f"{self.output_dir}/samed_raw"):
            image_files = sorted(glob(f"{self.output_dir}/samed_raw/{dataset_dir}/imagesTe/*.nii.gz"))
            label_files = sorted(glob(f"{self.output_dir}/samed_raw/{dataset_dir}/labelsTe/*.nii.gz"))

            print(f"🧾 Found {len(image_files)} images and {len(label_files)} labels in {dataset_dir}")

            pbar = tqdm(zip(image_files, label_files), total=len(image_files), desc=f"Creating H5 for {dataset_dir}", unit="volume")
            for image_file, label_file in pbar:
                basename = os.path.basename(image_file)
                match = re.search(r'img0(\d+)\.nii\.gz', basename)
                if not match:
                    print(f"⚠️ Nome file inatteso: {basename}")
                    continue
                number = match.group(1)

                try:
                    image_data = nib.load(image_file).get_fdata().astype(np.float32)
                    label_data = nib.load(label_file).get_fdata().astype(np.float32)
                except Exception as e:
                    print(f"❌ Errore nel caricamento: {e}")
                    continue

                image_data = np.clip(image_data, a_min, a_max)
                image_data = (image_data - a_min) / (a_max - a_min)
                image_data = np.transpose(image_data, (2, 1, 0))
                label_data = np.transpose(label_data, (2, 1, 0))

                for k, v in self.hashmap.items():
                    label_data[label_data == k] = v

                h5_path = os.path.join(output_dir, f"case{number}.npy.h5")
                with h5py.File(h5_path, 'w') as f:
                    f.create_dataset("image", data=image_data)
                    f.create_dataset("label", data=label_data)

            pbar.close()
                
    


if __name__ == "__main__":
    # Define the input and output directories
    input_data = "data/RETOUCH_dataset/RETOUCH"
    output_data = "data/RETOUCH_dataset/RETOUCH_PROCESSED_SAMED"

    # Create an instance of ProcessDataset and start the process
    processor = ProcessDataset(input_data, output_data)
    #processor.process_retouch_raw()
    #processor.preprocess_retouch_npz()
    #processor.createlists_retouch()
    processor.prepare_testset_h5() 
