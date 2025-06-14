from tqdm import tqdm
import os
import numpy as np
import nibabel as nib
import h5py
from glob import glob

hashmap = {
    0: 0,  # Background
    1: 1,  # Intraretinal fluid (IRF)
    2: 2,  # Subretinal fluid (SRF)
    3: 3   # Pigment Epithelial Detachment (PED)
}

def preprocess_valid_image(image_files: str, label_files: str) -> None:
    os.makedirs(f"test_vol_h5", exist_ok=True)

    a_min, a_max = -125, 275
    b_min, b_max = 0.0, 1.0

    pbar = tqdm(zip(image_files, label_files), total=len(image_files))
    for image_file, label_file in pbar:
        # **/imgXXXX.nii.gz -> parse XXXX
        number = image_file.split('/')[-1][3:7]


        image_data = nib.load(image_file).get_fdata()
        label_data = nib.load(label_file).get_fdata()

        image_data = image_data.astype(np.float32)
        label_data = label_data.astype(np.float32)

        image_data = np.clip(image_data, a_min, a_max)
        if True:
            image_data = (image_data - a_min) / (a_max - a_min)

        H, W, D = image_data.shape

        image_data = np.transpose(image_data, (2, 1, 0))
        label_data = np.transpose(label_data, (2, 1, 0))

        counter = 1
        for k in sorted(hashmap.keys()):
            counter += 1
            label_data[label_data == k] = hashmap[k]

        save_path = f"test_vol_h5/case{number}.npy.h5"
        f = h5py.File(save_path, 'w')
        f['image'] = image_data
        f['label'] = label_data
        f.close()
    pbar.close()
    
    
if __name__ == "__main__":
    
    image_folder = r"C:\Users\simone.roman\project\OCT_segmentation\data\RETOUCH_dataset\RETOUCH_PROCESSED_SAMED\samed_raw\Dataset003_Total\imagesTe"
    # read the files in the folder
    image_files = [f for f in glob(os.path.join(image_folder, "*.nii.gz"))]

    label_folder = r"C:\Users\simone.roman\project\OCT_segmentation\data\RETOUCH_dataset\RETOUCH_PROCESSED_SAMED\samed_raw\Dataset003_Total\labelsTe"
    label_files = [f for f in glob(os.path.join(label_folder, "*.nii.gz"))]
    preprocess_valid_image(image_files, label_files)
