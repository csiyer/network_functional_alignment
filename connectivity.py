""" 
This script calculates functional connectivity matrices for each session/subject from rest data.

Currently, this is implemented with parcels as our connectivity targets, i.e. the connectivity
matrices are (n_voxels x n_parcels).

It is also possible to implement this voxel-to-voxel (chose against this because of
computational demands + potential functional meaninglessness of voxel correlations) and with
searchlight spheres as our connectivity targets.

These connectomes are then passed to reliability.py to be assessed for within-subject reliability.
If they look good, we will average or concatenate them and then derive SRMs from them in srm.py.

This was run on Sherlock with 32 CPUs and 4G memory per CPU, which took 1-2 hours.

Author: Chris Iyer
Updated: 9/22/2023
"""

import glob
import numpy as np
import nibabel as nib
from nilearn import datasets
from nilearn.image import resample_img, math_img
from nilearn.maskers import NiftiMasker, NiftiLabelsMasker
from joblib import Parallel, delayed
from math import tanh 


def get_rest_filenames(local=False, BIDS_DIR = '/oak/stanford/groups/russpold/data/network_grant/discovery_BIDS_21.0.1/derivatives/glm_data_MNI'):
    if local:
        BIDS_DIR = 'data'
    files = glob.glob(BIDS_DIR + '/**/*rest*optcomDenoised*.nii.gz', recursive = True)
    confounds_files = glob.glob(BIDS_DIR + '/**/*rest*confounds*.tsv', recursive = True)
    return files, confounds_files

def shape_affine_checks(FILE_PATHS):
    """check if the shape & affine of every file match each other"""
    target_shape, target_affine = (nib.load(FILE_PATHS[0]).shape[0:3], nib.load(FILE_PATHS[0]).affine)
    for f in FILE_PATHS:
        img = nib.load(f)
        if img.shape[0:3] != target_shape or not np.all(img.affine == target_affine):
            print(f, img.shape[0:3], img.affine)
            return False
    return True

def get_gm_mask():
    """get gray matter mask (NOT CURRENTLY BEING USED, because parcellation is similar)"""
    gm_probseg = nib.load('data/templates/tpl-MNI152NLin2009cAsym_res-02_label-GM_probseg.nii.gz')
    return math_img('img >= 0.5', img=gm_probseg)

def get_parcellation(atlas = 'schaefer', n_dimensions = 400, resample_target=''):
    """get either Schaefer 2018 or DiFuMo parcellation. includes parcel labels, parcel map, and binarized mask of non-background voxels"""
    if atlas == 'schaefer':
        atlas = datasets.fetch_atlas_schaefer_2018(n_rois=n_dimensions, yeo_networks=7, resolution_mm=2, data_dir='data/templates', verbose=0)
    elif atlas == 'difumo':
        atlas = datasets.fetch_atlas_difumo(dimension=n_dimensions, resolution_mm=2, data_dir='data/templates', verbose=0)
    atlas_resampled = resample_img(atlas.maps, target_shape = resample_target.shape[:3], target_affine = resample_target.affine, interpolation = 'nearest')
    return atlas.labels, atlas_resampled, math_img('img > 0', img=atlas_resampled)

def get_combined_mask(local=False):
    # this is the combined gray matter + parcellation mask that we use to load in the data
    files, _ = get_rest_filenames(local) 
    gm_mask = get_gm_mask()
    _, _, parcel_mask = get_parcellation(atlas = 'schaefer', n_dimensions = 400, resample_target = nib.load(files[0])) 
    return math_img('img1 * img2', img1=gm_mask, img2=parcel_mask)


def load_data_one_session(FILE_PATH, CONFOUNDS_FILE = '', parcel_labels = [], parcel_map = []):
    """
    This function will load functional data for one subject/session from a given filename and confound name.
    It is called repeatedly on all the subjects. 

    Inputs:
        - FILE_PATH, CONFOUNDS_FILE: path names 
        - Parcel labels, map, and mask: outputs of get_parcellation above

    Outputs:
        - parcel_data: parcel-averaged timeseries values
        - voxel_data: voxel timeseries values, masked to all voxels within the gray matter mask + parcel map (so that we can track which parcel they're in later on)

    NOTE: our data has been formatted to the MNI152NLin2009cAsym_res-2 during fMRIPrep pre-processing
    NOTE: not using confounds, because instead using Tedana ICA denoising - revisit this?
    """
    combined_mask = get_combined_mask() # mask where it's gray matter above 50% and the parcellation applies

    masker_args = {
        'standardize': 'zscore_sample', # ?
        'n_jobs': 32,
    }

    voxel_masker = NiftiMasker(
        mask_img = combined_mask, 
        **masker_args
    )
    voxel_data = voxel_masker.fit_transform(FILE_PATH) # , confounds = CONFOUNDS_FILE) # see note above. using denoised files with no confounds

    parcel_masker = NiftiLabelsMasker(
        mask_img = combined_mask,
        labels_img = parcel_map,
        labels = parcel_labels,
        **masker_args
    )
    parcel_data = parcel_masker.fit_transform(FILE_PATH) # , confounds = CONFOUNDS_FILE)
    
    return voxel_data, parcel_data

def compute_fc_one_session(voxel_data, parcel_data, zscore = True): # TO-DO: MAKE THIS MORE EFFICIENT
    """
    Correlate each voxel's timeseries with each parcel's timeseries. Parallelized with joblib to speed it up.
    Inputs:
        - parcel_data and voxel_data for one session (from load_data_one_session) 
        - zscore: whether to Fisher z-transform (tanh) the correlation values)
    Outputs:
        - voxel-to-parcel connectivity matrix
    Notes:
        - Considered excluding the parcel in which each voxel resides from its correlation values, but then what to replace with? 
        - The parcel_data could be replaced with searchlight-averaged timeseries, or any other connectivity target.
        - Not using nilearn ConnectivityMeasure because this is across two matrices--couldn't figure that out
    """
    def correlate_one_voxel(i):
        corrs = np.corrcoef(voxel_data[:,i], parcel_data, rowvar=False)[0, 1:]
        if zscore:
            return [tanh(c) for c in corrs]
        return corrs

    list_of_voxels = Parallel(n_jobs = 32)(
        delayed(correlate_one_voxel)(i) for i in range(voxel_data.shape[1])
    )
    return np.vstack(list_of_voxels)


if __name__ == "__main__":

    files, confounds_files = get_rest_filenames() 
    print('Shape/affine checks result: ', shape_affine_checks(files))

    subject_session_list = [(f[f.find('sub'):f.find('sub')+7], f[f.find('ses'):f.find('ses')+6]) for f in files]
    print('subjects/sessions to do: \n', subject_session_list, '\n')

    parcel_labels, parcel_map, _ = get_parcellation(atlas = 'schaefer', n_dimensions = 400, resample_target = nib.load(files[0])) 

    for s,f,c in zip(subject_session_list, files, confounds_files): # actually not using confounds files in this iteration
        print('Beginning subject/session: ', s)

        voxel_data, parcel_data = load_data_one_session(f,c, parcel_labels, parcel_map)
        print('loaded data')
        
        connectome = compute_fc_one_session(voxel_data, parcel_data, zscore=True)
        print('got connectome')
        np.save(f'/scratch/users/csiyer/{s[0]}_{s[1]}_connectome.npy', connectome)

    print(f'finished all {len(files)} sessions!')
