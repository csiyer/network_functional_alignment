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
from nilearn.maskers import MultiNiftiMasker, MultiNiftiLabelsMasker
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from math import tanh 

def get_rest_filenames(BIDS_DIR):
    files = glob.glob(BIDS_DIR + '/**/*rest*optcomDenoised*.nii.gz', recursive = True)
    confounds_files = glob.glob(BIDS_DIR + '/**/*rest*confounds*.tsv', recursive = True)
    return files, confounds_files

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

def shape_affine_checks(FILE_PATHS):
    """check if the shape & affine of every file match each other"""
    target_shape, target_affine = (nib.load(FILE_PATHS[0]).shape[0:3], nib.load(FILE_PATHS[0]).affine)
    for f in FILE_PATHS:
        img = nib.load(f)
        if img.shape[0:3] != target_shape or not np.all(img.affine == target_affine):
            print(f, img.shape[0:3], img.affine)
            return False
    return True


def load_data(FILE_PATHS=[], CONFOUNDS_FILES = [], schaefer_n_rois=400,):
    """
    This function will load functional data from given files and return other information to help later on.

    Inputs:
        - FILE_PATHS: list of files with functional images to load (preferably all of one subject's sessions)
        - schaefer_n_rois: number of parcels of the Schaefer 2018 atlas that we load

    Outputs:
        - subject_session_list = list of (sub, ses) pairs that should match the data output
        - parcel_data: session-wise list of parcel-averaged timeseries values
        - voxel_data: session-wise list of voxel timeseries values
                Note that we're no longer keeping the voxel values parcel-organized when loading them in.
                Instead, we'll flatten the parcel map and use it to mask specific parcels later on.
        - parcel_map_flat: a *flattened* map of parcel values excluding the background. This should match the # of voxels in dimensions
                and correspond directly to what parcel each voxel belongs to.

    NOTE: our data has been formatted to the MNI152NLin2009cAsym_res-2 during fMRIPrep pre-processing
    """
    if FILE_PATHS == []:
        FILE_PATHS = glob.glob('data/rest/*.nii.gz') # all rest data in my current testing dir by default
    
    if not shape_affine_checks(FILE_PATHS):
        print('shape affine check error')
        return 'ERROR: data do not share the same shape and affine' 

    subject_session_list = [(f[f.find('sub'):f.find('sub')+7], f[f.find('ses'):f.find('ses')+6]) for f in FILE_PATHS]

    parcel_labels, parcel_map, parcel_mask = get_parcellation(schaefer_n_rois, resample_target = nib.load(FILE_PATHS[0])) 
    parcel_map_flat = parcel_map.get_fdata()[parcel_map.get_fdata() > 0].flatten()  
    np.save('../outputs/parcel_map_flat.npy', parcel_map_flat)

    masker_args = {
        'standardize': 'zscore_sample', # ?
        'n_jobs': -1,
    }

    parcel_masker = MultiNiftiLabelsMasker(
        labels_img = parcel_map,
        labels = parcel_labels,
        **masker_args
    )
    parcel_data = parcel_masker.fit_transform(FILE_PATHS, confounds = CONFOUNDS_FILES)
    print('loaded parcel data')

    voxel_masker = MultiNiftiMasker(
        mask_img = parcel_mask, # CRUCIAL: need this to be the case so we can know which parcel each voxel belongs to later
        **masker_args
    )
    voxel_data = voxel_masker.fit_transform(FILE_PATHS, confounds = CONFOUNDS_FILES)
    print('loaded voxel data')
    
    return subject_session_list, voxel_data, parcel_data, parcel_map_flat, parcel_labels


def compute_fcs(subject_session_list, voxel_data, parcel_data, zscore=True, save=True): # TO-DO: MAKE THIS MORE EFFICIENT
    """
    For each subject/session, we compute a functional connectivity matrix by correlating each voxel's timeseries with 
    each parcel's timeseries. We parallelize the process with joblib's Parallel and delayed() functions to make things quicker.
    
    We considered excluding the parcel in which each voxel resides from its correlation values, but then what do we replace the 
    value with in the correlation matrix? As is, nothing is excluded.

    Inputs:
        - (sub, ses) pairs describing each value in the data lists
        - parcel_data and voxel_data contain session-wise lists of either parcel-averaged or voxel values
        - zscore: whether to Fisher z-transform (tanh) the correlation values)
        - save: whether to save the connectomes to .npz files
    Outputs:
        - subject/session list of connectomes, matching the ordering of subject_session_list
    Notes:
        - The parcel_data could be replaced with searchlight-averaged timeseries, or any other connectivity target.
        - Not using nilearn ConnectivityMeasure because this is across two matrices--couldn't figure that out
    """
    print('voxel data shape: ', voxel_data[0].shape)
    print('parcel data shape: ', parcel_data[0].shape)
    connectomes = []

    def correlate_one_pair(v_col, p_col, zscore):
        corr = pearsonr(v_col, p_col)[0]
        if zscore:
            return tanh(corr)
        return corr

    def correlate_one_voxel(v_col, zscore):
        return [correlate_one_pair(v_col, p_col, zscore) for p_col in curr_parcel_data.T]
    
    print('starting connectomes')
    for curr_voxel_data, curr_parcel_data in zip(voxel_data, parcel_data):
        connectomes.append(
            Parallel(n_jobs=-1)(
                delayed(correlate_one_voxel)(v_col, zscore) for v_col in curr_voxel_data.T
            )
        )

    print('connectomes shape: ', connectomes[0].shape)
    print('saving connectomes')

    if save:
        np.save('../outputs/connectomes/subject_session_list.npy', subject_session_list)
        for s,c in zip(subject_session_list, connectomes):
            np.save(f'../outputs/connectomes/{s[0]}_{s[1]}_connectome.npy', c)

    return connectomes


if __name__ == "__main__":
    files, confounds_files = get_rest_filenames(BIDS_DIR = '/oak/stanford/groups/russpold/data/network_grant/discovery_BIDS_21.0.1/derivatives/glm_data_MNI') 
    subject_session_list, voxel_data, parcel_data, _, _ = load_data(files)
    # subject_session_list, voxel_data, parcel_data, _, _ = load_data(files, confounds_files) # parcel_data and voxel_data are lists of subjects
    print(f'loaded all {len(files)} files! \n')
    connectomes = compute_fcs(subject_session_list, voxel_data, parcel_data, zscore=True, save=True)
    