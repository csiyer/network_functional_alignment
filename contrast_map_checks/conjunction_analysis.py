"""
The decoding analysis is one way to quantify whether SRM seems to be picking up on between-subject signal 
in neural data during these control tasks. Another way to do this is to analyze GLM-derived subject-level 
contrasts directly, rather than trial-level decoding.

Here, we align each subject's GLM-derived contrast maps into shared space using the SRM transforms,
and compare across-subject alignment, in 2 ways:
    1) # of overlapping voxels in thresholded maps (SRM-transformed vs. not)
    2) Dice coefficient of unthresholded maps (SRM-transformed vs. not)

Author: Chris Iyer
Updated: 5/28/2024
"""
import os, sys, glob, json, itertools
import numpy as np
from sklearn.preprocessing import StandardScaler
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

def srm_transform(map, transform, zscore=True):
    out = np.dot(map, transform)
    if zscore:
        out = StandardScaler().fit_transform(out) 
    return out


def dice_coef(map1, map2):
    """
    Takes in two maps (masked into numpy arrays), binarizes them if necessary, and returns:
        1) correlation of maps
        2) dice coefficient of maps
    """
    # if isinstance(map1, nib.Nifti1Image) and isinstance(map2, nib.Nifti1Image):
    #     data1 = map1.get_fdata()
    #     data2 = map2.get_fdata()
    if isinstance(map1, np.ndarray) and isinstance(map2, np.ndarray):
        data1 = map1[0,:]
        data2 = map2[0,:]

    if data1.shape != data2.shape:
        raise ValueError("ERROR: shape mismatch")
    
    if not np.array_equal(np.unique(data1), [0,1]) or not np.array_equal(np.unique(data2), [0,1]):
        print('maps not thresholded, binarizing...')
        data1 = np.where(data1 > 0, 1, 0) # binarize
        data2 = np.where(data2 > 0, 1, 0)

    correlation, _ = pearsonr(data1, data2)

    intersection = np.sum(data1*data2)
    sum_binarized = np.sum(data1) + np.sum(data2)
    if sum_binarized == 0:
        return 1.0 if intersection == 0 else 0.0
    dice = 2.0 * intersection / sum_binarized

    return correlation, dice


def run_conjunction_analysis(save=True):
    """
    Main conjunction analysis script. Loops through contrast files, quantifies overlap and dice coefficients, 
    and saves output to json. 
    """
    output = {}
    for task in ['flanker','spatialTS','cuedTS','directedForgetting','stopSignal','goNogo', 'shapeMatching', 'nBack']:
        output[task] = {
            'srm': {'all_r': [], 'all_dice': []},
            'nosrm': {'all_r': [], 'all_dice': []}
        }

        subjects = np.unique(np.load(f'/scratch/users/csiyer/glm_outputs/{task}_subjects.npy'))
        full_task_fname = CONTRAST_PATH + f'{task}_lev1_output/task_{task}_rtmodel_rt_centered/contrast_estimates/'

        # create a data dictionary mapping subject names to both SRM transforms and contrast maps

        sub_dict = {sub: {
            'srm_transform': np.load([s for s in srm_files if sub in s][0]),
            'contrast_map': NiftiMasker(
                mask_img = get_combined_mask(), # mask where gray matter above 50% and the parcellation applies
                standardize = 'zscore_sample',
                n_jobs = -1
            ).fit_transform(glob.glob(full_task_fname + f'{sub}*{target_contrasts[task]}*t-test.nii.gz')[0])
        } for sub in subjects}

        for sub1, sub2 in itertools.combinations(subjects, 2): # for each possible pair, compare maps
            
            r, dice = dice_coef(sub_dict[sub1]['contrast_map'], sub_dict[sub2]['contrast_map'])
            output[task]['nosrm']['all_r'].append(r)
            output[task]['nosrm']['all_dice'].append(dice)

            r, dice = dice_coef(
                np.dot(sub_dict[sub1]['contrast_map'], sub_dict[sub1]['srm_transform']),
                np.dot(sub_dict[sub2]['contrast_map'], sub_dict[sub2]['srm_transform'])
            )

            output[task]['srm']['all_r'].append(r)
            output[task]['srm']['all_dice'].append(dice)

        for method in ['srm', 'nosrm']: # get averages
            output[task][method]['avg_r'] = np.mean(output[task][method]['all_r'])
            output[task][method]['avg_dice'] = np.mean(output[task][method]['all_dice'])
        
    if save: 
        OUTPATH = '/scratch/users/csiyer/conjunction_analysis/'
        if not os.path.isdir(OUTPATH):
            os.mkdir(OUTPATH)
        with open(OUTPATH + 'outputs.json', 'w') as file:
            json.dump(output, file, indent=4)

    return results


def plot_results(results, save=True):
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
        plt.savefig('/scratch/users/csiyer/conjunction_analysis/results.png')
    else:
        plt.show()


if __name__ == '__main__':
    results = run_conjunction_analysis()
    plot_results(results)
    