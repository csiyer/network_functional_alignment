""" 
This script calculates functional connectivity matrices for each session/subject from rest data.

Currently, this is implemented with parcels as our connectivity targets, i.e. the connectivity
matrices are (n_voxels x n_parcels).

It is also possible to implement this voxel-to-voxel (chose against this because of
computational demands + potential functional meaninglessness of voxel correlations) and with
searchlight spheres as our connectivity targets.

These connectomes are then passed to reliability.py to be assessed for within-subject reliability.
If they look good, we will average or concatenate them and then derive SRMs from them in srm.py.

Author: Chris Iyer
Updated: 8/11/2023
"""

import glob
import numpy as np
import nibabel as nib
from nilearn import datasets
from nilearn.image import resample_img, math_img
from nilearn.maskers import NiftiMasker, NiftiLabelsMasker
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from math import tanh 

def get_rest_filenames(BIDS_DIR):
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

def get_parcellation(schaefer_n_rois=400, resample_target=''):
    """get Schaefer 2018 parcellation. includes parcel labels, parcel map, and binarized mask of non-background voxels"""
    schaefer_atlas = datasets.fetch_atlas_schaefer_2018(n_rois=schaefer_n_rois, 
                                                        yeo_networks=7, 
                                                        resolution_mm=2,
                                                        data_dir='data/templates', 
                                                        verbose=0)
    schaefer_resampled = resample_img(schaefer_atlas.maps, 
                                        target_shape = resample_target.shape[:3],
                                        target_affine = resample_target.affine,
                                        interpolation = 'nearest')

    return schaefer_atlas.labels, schaefer_resampled, math_img('img > 0', img=schaefer_resampled)


def load_data_one_session(FILE_PATH, CONFOUNDS_FILE = '', parcel_labels = [], parcel_map = [], parcel_mask = []):
    """
    This function will load functional data for one subject/session from a given filename and confound name.
    It is called repeatedly on all the subjects. 

    Inputs:
        - FILE_PATH, CONFOUNDS_FILE: path names 
        - Parcel labels, map, and mask: outputs of get_parcellation above

    Outputs:
        - parcel_data: parcel-averaged timeseries values
        - voxel_data: voxel timeseries values, masked to all voxels within the parcel map (so that we can track which parcel they're in later on)

    NOTE: our data has been formatted to the MNI152NLin2009cAsym_res-2 during fMRIPrep pre-processing
    """
    masker_args = {
        'standardize': 'zscore_sample', # ?
        'n_jobs': 40,
    }

    voxel_masker = NiftiMasker(
        mask_img = parcel_mask, # CRUCIAL: need this to be the case so we can know which parcel each voxel belongs to later
        **masker_args
    )
    voxel_data = voxel_masker.fit_transform(FILE_PATH, confounds = CONFOUNDS_FILE)

    parcel_masker = NiftiLabelsMasker(
        labels_img = parcel_map,
        labels = parcel_labels,
        **masker_args
    )
    parcel_data = parcel_masker.fit_transform(FILE_PATH, confounds = CONFOUNDS_FILE)
    
    return voxel_data, parcel_data

def compute_fc_one_session(voxel_data, parcel_data, zscore = True): # TO-DO: MAKE THIS MORE EFFICIENT
    """
    Correlate each voxel's timeseries with each parcel's timeseries. Parallelized with joblib to speed it up.
    Inputs:
        - parcel_data and voxel_data for one session (from load_data_one_session) 
        - zscore: whether to Fisher z-transform (tanh) the correlation values)
    Outputs:
        - connectome
    Notes:
        - Considered excluding the parcel in which each voxel resides from its correlation values, but then what to replace with? 
        - The parcel_data could be replaced with searchlight-averaged timeseries, or any other connectivity target.
        - Not using nilearn ConnectivityMeasure because this is across two matrices--couldn't figure that out
    """
    def correlate_one_pair(v_col, p_col, zscore):
        corr = pearsonr(v_col, p_col)[0]
        if zscore:
            return tanh(corr)
        return corr

    def correlate_one_voxel(v_col, zscore):
        return [correlate_one_pair(v_col, p_col, zscore) for p_col in parcel_data.T]
    
    list_of_voxels = Parallel(n_jobs=40)(
        delayed(correlate_one_voxel)(v_col, zscore) for v_col in voxel_data.T
    )

    return np.vstack(list_of_voxels)


if __name__ == "__main__":

    files, confounds_files = get_rest_filenames(BIDS_DIR = '/oak/stanford/groups/russpold/data/network_grant/discovery_BIDS_21.0.1/derivatives/glm_data_MNI') 
    
    subject_session_list = [(f[f.find('sub'):f.find('sub')+7], f[f.find('ses'):f.find('ses')+6]) for f in files]
    print('Shape/affine checks result: ', shape_affine_checks(files))
    parcel_labels, parcel_map, parcel_mask = get_parcellation(schaefer_n_rois = 400, resample_target = nib.load(files[0])) 
    parcel_map_flat = parcel_map.get_fdata()[parcel_map.get_fdata() > 0].flatten()  
    # np.save('../outputs/parcel_map_flat.npy', parcel_map_flat)

    for s,f,c in zip(subject_session_list, files, confounds_files):
        print('Beginning subject/session: ', s)

        voxel_data, parcel_data = load_data_one_session(f,c, parcel_labels, parcel_map, parcel_mask)
        print('loaded data')
        connectome = compute_fc_one_session(voxel_data, parcel_data, zscore=True)
        np.save(f'../outputs/connectomes/{s[0]}_{s[1]}_connectome.npy', connectome)

    print(f'finished all {len(files)} sessions!')
