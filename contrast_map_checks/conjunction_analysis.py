"""
The decoding analysis is one way to quantify whether SRM seems to be picking up on between-subject signal 
in neural data during these control tasks. Another way to do this is to analyze GLM-derived subject-level 
contrasts directly, rather than trial-level decoding.

This script transforms GLM-derived contrast maps into one target subject's native space, and compares their
across-subject overlap to that of non-transformed contrast maps. 

Because of concerns about data leakage when using target subjects, this script re-derives SRM matrices 
excluding one subject, adds the left-out subject to the shared model,
and then transforms each subject's maps into the left-out subject's native space.

Finally, overlap is quantified by ____?

https://github.com/neurodatascience/fmralign-benchmark/blob/5e06751a2dfab7d8c0494dabce32a2e0aac31615/experiments/experiment_3.py

Author: Chris Iyer
Updated: 6/5/2024
"""

import os, sys, glob, json, itertools
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from nilearn.maskers import NiftiMasker
from joblib import Parallel, delayed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from connectivity import get_combined_mask
from srm import load_parcel_map
from utils.fastsrm_brainiak import FastSRM 


def load_all_filenames():
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
    sub_files = {}
    connectome_files = sorted(glob.glob('/scratch/users/csiyer/connectomes/*avg*'))
    sub_list = [f[f.find('sub')+4:f.find('sub')+7] for f in connectome_files] # s03, s10, etc.

    for s,c in zip(sub_list, connectome_files):
        sub_files[s] = {'connectome': c}
        for task in ['flanker','spatialTS','cuedTS','directedForgetting','stopSignal','goNogo', 'shapeMatching', 'nBack']:
            CONTRAST_PATH = '/oak/stanford/groups/russpold/data/network_grant/discovery_BIDS_21.0.1/derivatives/output_optcom_MNI/'
            full_task_fname = CONTRAST_PATH + f'{task}_lev1_output/task_{task}_rtmodel_rt_centered/contrast_estimates/'
            sub_files[s][task] = glob.glob(full_task_fname + f'sub-{s}*{target_contrasts[task]}*t-test.nii.gz')[0]

    return sub_files


def mask(img_path):
    # return image.binarize_img(img, threshold=threshold_val, mask_img=get_combined_mask(local=True))
    return NiftiMasker(
        mask_img = get_combined_mask(), # mask where gray matter above 50% and the parcellation applies
        n_jobs = -1
    ).fit_transform(img_path)[0,:] # just first slice (contrast maps are 2D)


def srm_and_loso_native(map, sub_transform, s1_transform):
    """Accepts a contrast map and which subject it belongs to, and dictionary of data. 
    Transforms the map into shared space and then back into the first subject's native space."""
    map_to_shared = np.dot(map, sub_transform)
    return np.dot(map_to_shared, s1_transform.T)


def compute_loso_srm(data_list, sub_list, loso_sub, parcel_map, n_features=100, save=True):
    """
    This function uses BrainIAK's Shared Response Modeling function to compute parcel-wise SRMs (one per parcel, as an anatomical constraint).
    CRUCIALLY, the shared model is derived on all subjects but one; that left-out subject is subsequently added to derive their own transformation matrix.
    This is done for the purpose of eliminating data leakage when we later transform all other subjects' data into that subject's native space.
    
    Inputs: list of connectomes, corresponding subject names, which subject to leave out, dimension-matched parcel map, shared model # dimensions, save?
    Output: transformation matrices for each subject, in the order of sub_list and data_list inputs

    NOTE: the parcelwise shared responses are not saved here (just the transformation matrices), because I haven't been using them for anything.
    """
    outpath = os.path.join('/scratch/users/csiyer/srm_outputs/', f'loso/{loso_sub}')
    if os.path.exists(outpath): # if this script has been run before, we can load past results instead of re-deriving
        return [np.load(glob.glob(f'{outpath}/sub-{sub}_srm_transform_loso-{loso_sub}.npy')[0]) for sub in sub_list]
    else:
        os.makedirs(outpath)

    loso_idx = sub_list.index(loso_sub)
    loso_data = data_list[loso_idx]
    train_data = [d for i,d in enumerate(data_list) if i != loso_idx]

    temp_dir = '/scratch/users/csiyer/temp_dir'

    def single_parcel_srm(train_data, loso_data, parcel_map, parcel_label, n_features):
        parcel_idxs = np.where(parcel_map == parcel_label)[0]
        train_data_parcel = [d[parcel_idxs] for d in train_data]
        srm = FastSRM(n_components=n_features, n_iter=20, n_jobs=1, aggregate='mean', temp_dir = temp_dir)
        reduced_sr = srm.fit_transform(train_data_parcel)
        srm.aggregate = None
        srm.add_subjects([loso_data[parcel_idxs]], reduced_sr)
        return [np.load(x).T for x in srm.basis_list], parcel_idxs # return list of all the transforms (transpose of basis), which are saved to temp_dir

    srm_outputs = Parallel(n_jobs=-1)(
        delayed(single_parcel_srm)(train_data, loso_data, parcel_map, parcel_label, n_features) for parcel_label in np.sort(np.unique(parcel_map))
    )

    subject_transforms = [np.zeros((len(parcel_map), n_features)) for _ in range(len(sub_list))] # empty initalize
    for sub_weights, parcel_idxs in srm_outputs: # for each parcel
        for i,sub in enumerate(subject_transforms): # for each subject
            sub[parcel_idxs,:] = sub_weights[i] # add those parcel's transformation values to that subject's transformation matrix

    # we need to reorder them to match the original sub_list. the loso_sub's transform is at the end, and we'll insert it at its original index
    subject_transforms.insert(loso_idx, subject_transforms.pop())

    if save:
        for sub, transform in zip(sub_list, subject_transforms):
            np.save(f'{outpath}/sub-{sub}_srm_transform_loso-{loso_sub}.npy', transform)

    return subject_transforms


def dice_coef(map1, map2, threshold_val=2):
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


def run_conjunction_analysis(srm_n_features=100, threshold_val=2, save=True, savetag=''):
    """
    Main conjunction analysis script. 
    Loops through the subjects, for each one:
        - derives SRM transforms into that subject's native space (leaving that subject out from the initial SRM)
        - transforms contrast maps into that native space
        - quantify between-subject overlap in both transformed and un-transformed maps
    plots output and saves results to .json file
    """
    sub_files = load_all_filenames() # sub_files['s3']['connectome'] = <path>;  sub_files['s3']['flanker'] = <path>
    parcel_map = load_parcel_map(n_dimensions = 100)
    results = {} 
    
    for task in ['flanker','spatialTS','cuedTS','directedForgetting','stopSignal','goNogo', 'shapeMatching', 'nBack']:
        print(f'starting {task}')
        results[task] = {
            'srm': {'r': [], 'dice': []}, 
            'no_srm': {'r': [], 'dice': []}
        }

        for loso_idx,loso_sub in enumerate(sub_files.keys()): # leave one subject out of shared model creation, transform maps into their native space
            print(f'loso round {loso_idx+1} of {len(sub_files.keys())}')

            sub_list = list(sub_files.keys())
            data_list = [np.load(v['connectome']) for s,v in sub_files.items()]

            sub_transforms = compute_loso_srm(data_list, sub_list, loso_sub, parcel_map, n_features=srm_n_features, save=save)
            contrast_maps = [mask(sub_files[s][task]) for s in sub_list]
            contrast_maps_srm = [
                m if s == loso_sub else srm_and_loso_native(m, t, sub_transforms[loso_idx])
                for s, m, t in zip(sub_list, contrast_maps, sub_transforms)
            ]

            # now we have contrast maps that are both transformed and untransformed, and just need to quantify overlap for both
            # to do so we will get the average dice coefficient between every possible pair of maps
            pairs = itertools.combinations(range(len(sub_list)), 2)
            results[task]['no_srm']['dice'].append(
                [dice_coef(contrast_maps[i], contrast_maps[j], threshold_val = threshold_val) for i,j in pairs]
            )
            results[task]['srm']['dice'].append(
                [dice_coef(contrast_maps_srm[i], contrast_maps_srm[j], threshold_val = threshold_val) for i,j in pairs]
            )
            results[task]['no_srm']['r'].append(
                [pearsonr(contrast_maps[i], contrast_maps[j])[0] for i,j in pairs]
            )
            results[task]['srm']['r'].append(
                [pearsonr(contrast_maps_srm[i], contrast_maps_srm[j])[0] for i,j in pairs]
            )

        # now, we want to average across the left-out subjects
        for method, measure in itertools.product(['srm','no_srm'], ['dice', 'r']):
            results[task][method][measure] = np.mean(results[task][method][measure], axis=0) 

    if save: 
        OUTPATH = '/scratch/users/csiyer/conjunction_analysis/'
        if not os.path.isdir(OUTPATH):
            os.mkdir(OUTPATH)
        with open(OUTPATH + f'results_{savetag}.json', 'w') as file:
            json.dump(results, file, indent=4)
        print('saved results!')

    return results


def plot_results(results, save=True, savetag=''):
    tasks = list(results.keys())
    fig, axes = plt.subplots(2,1, figsize=(8, 8))
    
    # Subplot for Correlations
    axes[0].set_title("Contrast Map Correlation")
    axes[0].set_ylabel("Average Pearson r (across pairs of subjects)")
    axes[0].set_xticks(range(len(tasks)))
    axes[0].set_xticklabels(tasks, rotation=45, ha='right')
    
    # Subplot for Dice Coefficients
    axes[1].set_title("Contrast Map Dice Coefficients")
    axes[1].set_ylabel("Average Dice Coefficient (across pairs of subjects)")
    axes[1].set_xticks(range(len(tasks)))
    axes[1].set_xticklabels(tasks, rotation=45, ha='right')
    
    # Plot data
    for i, task in enumerate(tasks):
        axes[0].bar(i - 0.2, np.mean(results[task]['srm']['r']), yerr=np.std(results[task]['srm']['r']), 
                    width=0.4, label='SRM' if i == 0 else "", color='green', capsize=5, alpha = 0.5)
        axes[0].bar(i + 0.2, np.mean(results[task]['no_srm']['r']), yerr=np.std(results[task]['no_srm']['r']), 
                    width=0.4, label='No SRM' if i == 0 else "", color='blue', capsize=5, alpha = 0.5)
        
        axes[1].bar(i - 0.2, np.mean(results[task]['srm']['dice']), yerr=np.std(results[task]['srm']['dice']), 
                    width=0.4, label='SRM' if i == 0 else "", color='green', capsize=5, alpha = 0.5)
        axes[1].bar(i + 0.2, np.mean(results[task]['no_srm']['dice']), yerr=np.std(results[task]['no_srm']['dice']), 
                    width=0.4, label='No SRM' if i == 0 else "", color='blue', capsize=5, alpha = 0.5)

    axes[0].legend()
    axes[1].legend()
    plt.tight_layout()
    
    if save:
        plt.savefig(f'/scratch/users/csiyer/conjunction_analysis/results_{savetag}.png')
    else:
        plt.show()
    print('finished plotting!')


if __name__ == '__main__':
    results = run_conjunction_analysis(srm_n_features=100, threshold_val=2, save=True, savetag = 'loso_version')
    plot_results(results)
    