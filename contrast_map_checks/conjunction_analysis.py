"""
The decoding analysis is one way to quantify whether SRM seems to be picking up on between-subject signal 
in neural data during these control tasks. Another way to do this is to analyze GLM-derived subject-level 
contrasts directly, rather than trial-level decoding.

Here, we align each subject's GLM-derived contrast maps into shared space using the SRM transforms,
and compare across-subject alignment, in 2 ways:
    1) # of overlapping voxels in thresholded maps (SRM-transformed vs. not)
    2) Dice coefficient of unthresholded maps (SRM-transformed vs. not)

Author: Chris Iyer
Updated: 6/5/2024
"""

import os, sys, glob, json, itertools
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from nilearn.maskers import NiftiMasker

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from connectivity import get_combined_mask


CONTRAST_PATH = '/oak/stanford/groups/russpold/data/network_grant/discovery_BIDS_21.0.1/derivatives/output_optcom_MNI/'
SRM_DIR = '/scratch/users/csiyer/srm_outputs/'
srm_files = glob.glob(SRM_DIR + '*transform*')

target_contrasts = {
    'cuedTS': 'cuedTS_contrast-task_switch_cost',
    'directedForgetting': 'directedForgetting_contrast-neg-con',
    'flanker': 'flanker_contrast-incongruent - congruent',
    'nBack': 'nBack_contrast-twoBack-oneBack',
    'spatialTS': 'spatialTS_contrast-task_switch_cost',
    'shapeMatching': 'shapeMatching_contrast-main_vars',
    'goNogo': 'goNogo_contrast-nogo_success-go',
    'stopSignal': 'stopSignal_contrast-stop_failure-go'
}


def mask(img_path):
    # return image.binarize_img(img, threshold=threshold_val, mask_img=get_combined_mask(local=True))
    return NiftiMasker(
        mask_img = get_combined_mask(), # mask where gray matter above 50% and the parcellation applies
        n_jobs = -1
    ).fit_transform(img_path)[0,:] # just first slice (contrast maps are 2D)


def srm_and_s1_native(map, sub_transform, s1_transform):
    """Accepts a contrast map and which subject it belongs to, and dictionary of data. 
    Transforms the map into shared space and then back into the first subject's native space."""
    srm_data = np.dot(map, sub_transform)
    return np.dot(srm_data, s1_transform.T)


def dice_coef(map1, map2, threshold_val = 2):
    """
    Takes in two maps (masked into numpy arrays), and returns:
        1) correlation of unthresholded maps
        2) dice coefficient of binarized/thresholded maps
    """
    if map1.shape != map2.shape:
        raise ValueError("ERROR: shape mismatch")
    if np.array_equal(np.unique(map1), [0,1]) or np.array_equal(np.unique(map2), [0,1]):
        raise ValueError("ERROR: maps already binarized?")
    # if isinstance(map1, nib.Nifti1Image) and isinstance(map2, nib.Nifti1Image):
    #     data1 = map1.get_fdata()
    #     data2 = map2.get_fdata()
    # if isinstance(map1, np.ndarray) and isinstance(map2, np.ndarray):
    #     data1 = map1[0,:]
    #     data2 = map2[0,:]

    # binarize
    map1 = np.where(map1 > threshold_val, 1, 0) # binarize
    map2 = np.where(map2 > threshold_val, 1, 0)

    intersection = np.sum(map1*map2)
    sum_binarized = np.sum(map1) + np.sum(map2)
    if sum_binarized == 0:
        return 1.0 if intersection == 0 else 0.0
    dice = 2.0 * intersection / sum_binarized

    return dice


def run_conjunction_analysis(threshold_val = 2, save=True):
    """
    Main conjunction analysis script. Loops through contrast files, quantifies overlap and dice coefficients, 
    and saves output to json. 
    """
    results = {}
    for task in ['flanker','spatialTS','cuedTS','directedForgetting','stopSignal','goNogo', 'shapeMatching', 'nBack']:
        results[task] = {
            'srm': {'all_r': [], 'all_dice': []},
            'nosrm': {'all_r': [], 'all_dice': []}
        }

        subjects = np.unique(np.load(f'/scratch/users/csiyer/glm_outputs/{task}_subjects.npy'))
        full_task_fname = CONTRAST_PATH + f'{task}_lev1_output/task_{task}_rtmodel_rt_centered/contrast_estimates/'

        # construct dictionary of all maps to be compared
        sub_dict = {}
        s1_transform = np.load(sorted(srm_files)[0]) # first subject's srm transform, to put all SRM'd data back in this native space
        for sub in subjects:
            contrast_map = mask(glob.glob(full_task_fname + f'{sub}*{target_contrasts[task]}*t-test.nii.gz')[0])
            srm_transform = np.load([s for s in srm_files if sub in s][0])
            sub_dict[sub] = {
                'contrast_map': contrast_map,
                'contrast_map_srm': srm_and_s1_native(contrast_map, srm_transform, s1_transform)
            }

        for sub1, sub2 in itertools.combinations(subjects, 2): # for each possible pair, compare maps

            ### No SRM version
            dice = dice_coef(sub_dict[sub1]['contrast_map'], sub_dict[sub2]['contrast_map'], threshold_val = threshold_val)
            r, _ = pearsonr(sub_dict[sub1]['contrast_map'], sub_dict[sub2]['contrast_map'])
            results[task]['nosrm']['all_r'].append(r)
            results[task]['nosrm']['all_dice'].append(dice)

            ### SRM version
            dice = dice_coef(sub_dict[sub1]['contrast_map_srm'], sub_dict[sub2]['contrast_map_srm'], threshold_val = threshold_val)
            r, _ = pearsonr(sub_dict[sub1]['contrast_map_srm'], sub_dict[sub2]['contrast_map_srm'])
            results[task]['srm']['all_r'].append(r)
            results[task]['srm']['all_dice'].append(dice)

        # average results
        for method in ['srm', 'nosrm']: # get averages
            results[task][method]['avg_r'] = np.mean(results[task][method]['all_r'])
            results[task][method]['avg_dice'] = np.mean(results[task][method]['all_dice'])
        
    if save: 
        OUTPATH = '/scratch/users/csiyer/conjunction_analysis/'
        if not os.path.isdir(OUTPATH):
            os.mkdir(OUTPATH)
        with open(OUTPATH + 'results.json', 'w') as file:
            json.dump(results, file, indent=4)

    return results


def plot_results(results, save=True, savetag = ''):
    tasks = list(results.keys())
    fig, axes = plt.subplots(2,1, figsize=(8, 8))
    
    # Subplot for Correlations
    axes[0].set_title("Contrast Map Correlation")
    axes[0].set_ylabel("Average Pearson r")
    axes[0].set_xticks(range(len(tasks)))
    axes[0].set_xticklabels(tasks, rotation=45, ha='right')
    
    # Subplot for Dice Coefficients
    axes[1].set_title("Contrast Map Dice Coefficients")
    axes[1].set_ylabel("Average Dice Coefficient")
    axes[1].set_xticks(range(len(tasks)))
    axes[1].set_xticklabels(tasks, rotation=45, ha='right')
    
    # Plot data
    for i, task in enumerate(tasks):
        axes[0].bar(i - 0.2, results[task]['srm']['avg_r'], yerr=np.std(results[task]['srm']['all_r']), 
                    width=0.4, label='SRM' if i == 0 else "", color='green', capsize=5, alpha = 0.5)
        axes[0].bar(i + 0.2, results[task]['nosrm']['avg_r'], yerr=np.std(results[task]['nosrm']['all_r']), 
                    width=0.4, label='No SRM' if i == 0 else "", color='blue', capsize=5, alpha = 0.5)
        
        axes[1].bar(i - 0.2, results[task]['srm']['avg_dice'], yerr=np.std(results[task]['srm']['all_dice']), 
                    width=0.4, label='SRM' if i == 0 else "", color='green', capsize=5, alpha = 0.5)
        axes[1].bar(i + 0.2, results[task]['nosrm']['avg_dice'], yerr=np.std(results[task]['nosrm']['all_dice']), 
                    width=0.4, label='No SRM' if i == 0 else "", color='blue', capsize=5, alpha = 0.5)

    axes[0].legend()
    axes[1].legend()
    plt.tight_layout()
    
    if save:
        plt.savefig(f'/scratch/users/csiyer/conjunction_analysis/results_{savetag}.png')
    else:
        plt.show()


if __name__ == '__main__':
    results = run_conjunction_analysis()
    plot_results(results)
    