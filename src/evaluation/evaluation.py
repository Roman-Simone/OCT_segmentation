import multiprocessing
import os
import json
from copy import deepcopy
from multiprocessing import Pool
from typing import Tuple, List, Union, Optional

import numpy as np
import SimpleITK as sitk

class Evaluation:
    
    def __init__(self, folder_ref: str, folder_pred: str, output_file: str, file_ending: str, regions: List[int], ignore_label: int = None, num_processes: int = 8):
        self.folder_ref = folder_ref
        self.folder_pred = folder_pred
        self.output_file = output_file
        self.file_ending = file_ending
        self.regions = regions
        self.ignore_label = ignore_label
        self.num_processes = num_processes
    

    def label_or_region_to_key(self, label_or_region: Union[int, Tuple[int]]):
        return str(label_or_region)

    def key_to_label_or_region(self, key: str):
        try:
            return int(key)
        except ValueError:
            key = key.replace('(', '').replace(')', '')
            split = key.split(',')
            return tuple([int(i) for i in split if len(i) > 0])

    def save_json(self, data: dict, filename: str, sort_keys: bool = True):
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj
        
        data = convert_types(data)
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4, sort_keys=sort_keys)
            
    
    def load_json(self, filename: str):
        with open(filename, 'r') as f:
            return json.load(f)

    def labels_to_list_of_regions(self, labels: List[int]):
        return [(i,) for i in labels]

    def region_or_label_to_mask(self,segmentation: np.ndarray, region_or_label: Union[int, Tuple[int, ...]]) -> np.ndarray:
        if np.isscalar(region_or_label):
            return segmentation == region_or_label
        else:
            mask = np.zeros_like(segmentation, dtype=bool)
            for r in region_or_label:
                mask[segmentation == r] = True
        return mask

    def compute_tp_fp_fn_tn(self, mask_ref: np.ndarray, mask_pred: np.ndarray, ignore_mask: np.ndarray = None):
        if ignore_mask is None:
            use_mask = np.ones_like(mask_ref, dtype=bool)
        else:
            use_mask = ~ignore_mask
        tp = np.sum((mask_ref & mask_pred) & use_mask)
        fp = np.sum(((~mask_ref) & mask_pred) & use_mask)
        fn = np.sum((mask_ref & (~mask_pred)) & use_mask)
        tn = np.sum(((~mask_ref) & (~mask_pred)) & use_mask)
        return tp, fp, fn, tn

    def compute_metrics(self, reference_file: str, prediction_file: str, labels_or_regions: Union[List[int], List[Union[int, Tuple[int, ...]]]], ignore_label: int = None) -> dict:
        
        seg_ref = sitk.GetArrayFromImage(sitk.ReadImage(reference_file))
        seg_pred = sitk.GetArrayFromImage(sitk.ReadImage(prediction_file))
        ignore_mask = seg_ref == ignore_label if ignore_label is not None else None
        
        
        results = {'reference_file': reference_file, 'prediction_file': prediction_file, 'metrics': {}}
        for r in labels_or_regions:
            key = self.label_or_region_to_key(r)
            results['metrics'][key] = {}
            mask_ref = self.region_or_label_to_mask(seg_ref, r)
            mask_pred = self.region_or_label_to_mask(seg_pred, r)
            tp, fp, fn, tn = self.compute_tp_fp_fn_tn(mask_ref, mask_pred, ignore_mask)
            results['metrics'][key]['Dice'] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else np.nan
            results['metrics'][key]['IoU'] = tp / (tp + fp + fn) if tp + fp + fn > 0 else np.nan
            results['metrics'][key].update({'FP': fp, 'TP': tp, 'FN': fn, 'TN': tn, 'n_pred': fp + tp, 'n_ref': fn + tp})
            results['metrics'][key]['f1'] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else np.nan 
        return results

    def compute_metrics_on_folder(self, folder_ref: str, folder_pred: str, output_file: str, file_ending: str, regions_or_labels: Union[List[int], List[Union[int, Tuple[int, ...]]]], ignore_label: int = None, num_processes: int = 8, chill: bool = False) -> dict:
        files_pred = [f for f in os.listdir(folder_pred) if f.endswith(file_ending)]
        files_ref = [f for f in os.listdir(folder_ref) if f.endswith(file_ending)]
        # if not chill:
        #     assert all(os.path.isfile(os.path.join(folder_pred, f)) for f in files_ref), "Not all files in folder_ref exist in folder_pred"
        files_ref = [os.path.join(folder_ref, f) for f in files_pred]
        files_pred = [os.path.join(folder_pred, f) for f in files_pred]
        
        with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
            results = pool.starmap(self.compute_metrics, zip(files_ref, files_pred, [regions_or_labels] * len(files_pred), [ignore_label] * len(files_pred)))
        
        metric_list = list(results[0]['metrics'][self.label_or_region_to_key(regions_or_labels[0])].keys())
        means = {self.label_or_region_to_key(r): {m: np.nanmean([i['metrics'][self.label_or_region_to_key(r)][m] for i in results]) for m in metric_list} for r in regions_or_labels}
        foreground_mean = {m: np.mean([means[k][m] for k in means.keys() if k != '0']) for m in metric_list}
        
        result = {'metric_per_case': results, 'mean': means, 'foreground_mean': foreground_mean}
        if output_file is not None:
            self.save_json(result, output_file)
        return result

def main():
    folder_ref = '/Users/simoneroman/Desktop/OCT_segmentation/data/RETOUCH_dataset/RETOUCH_PROCESSED_SAMED/prediction/predictions/ref'
    folder_pred = '/Users/simoneroman/Desktop/OCT_segmentation/data/RETOUCH_dataset/RETOUCH_PROCESSED_SAMED/prediction/predictions/pred'
    output_file = 'summary2.json'
    file_ending = '.nii.gz'
    regions = [0, 1, 2]
    ignore_label = None
    num_processes = 8
    eval = Evaluation(folder_ref, folder_pred, output_file, file_ending, regions, ignore_label, num_processes)
    eval.compute_metrics_on_folder(folder_ref, folder_pred, output_file, file_ending, regions, ignore_label, num_processes)

if __name__ == "__main__":
    main()
