""" This script will use functions from data_io_functions and fc_functions to 
calculate functional connectivity matrices for each session/subject from rest data.

These can be implemented in a few ways: voxel-to-voxel connectivity (correlate
each voxel to all others), voxel-to-parcel (correlate each voxel to each parcel
it's not within), and voxel-to-searchlight (correlate each voxel to each searchlight
tiling the cortex that it's not within). I think voxel-to-parcel (i.e., parcels are our 
"connectivity targets") is most promising.

We will then need to figure out how to combine these connectomes within-subject,
across-session (e.g., average them, concatenate their timeseries and recalculate).
Our strategy will depend on their within-subject reliability, assessed in connectome_reliability.py

We will then compute SRMs based on these connectomes in srm.py if they look good!

Author: Chris Iyer
Updated: 7/13/2023
"""

import glob
import numpy as np
from nilearn.maskers import MultiNiftiMasker, MultiNiftiLabelsMasker
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure

def load_all_data():
    # loads in nifti data by extracting voxel timeseries

    FILE_PATHS = glob.glob('data/*rest*.nii.gz')
    CONFOUND_PATHS = glob.glob('data/*confounds*.tsv')
    masker_args = {
        'standardize': 'zscore_sample', # ??
        'mask_strategy': 'gm-template',
        'n_jobs': -1
        # not doing any: smoothing, detrend, standardize, low_pass, high_pass, t_r
    }
    masker = MultiNiftiMasker(**masker_args)
    brain_data = masker.fit_transform(FILE_PATHS, confounds=CONFOUND_PATHS)
    return brain_data
    
def load_all_parcels(schaefer_n_rois=400):
    # loads in nifti data by extracting parcel timeseries

    FILE_PATHS = glob.glob('data/*rest*.nii.gz')
    CONFOUND_PATHS = glob.glob('data/*confounds*.tsv')
    schaefer_atlas = datasets.fetch_atlas_schaefer_2018(n_rois=schaefer_n_rois, 
                                                        yeo_networks=7, 
                                                        resolution_mm=1, 
                                                        data_dir='data/schaefer', 
                                                        verbose=0)
    masker_args = {
        'standardize': 'zscore_sample', # ??
        'labels_img': schaefer_atlas.maps,
        'n_jobs': -1,
        'strategy': 'mean'
        # not doing any: smoothing, detrend, standardize, low_pass, high_pass, t_r
    }
    masker = MultiNiftiLabelsMasker(**masker_args)
    parcels_data = masker.fit_transform(FILE_PATHS, confounds=CONFOUND_PATHS)
    return parcels_data

def compute_fc_voxel():
    # voxel-to-voxel connectivity/correlation matrix (i.e., FCMA)
    # uses brainiak fcma functions for efficiency!
    pass

def compute_fc_parcel():
    correlation_measure = ConnectivityMeasure(kind="correlation")
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]
    return correlation_matrix

def compute_fc_searchlight():
    # haven't figured out how to write this function yet, but it's intended to
    # replace parcels with searchlights as the connectivity targets. Soon!
    pass

def write_connectomes():
    # write our outputs back to files 

if __name__ == "__main__":
    brain_data = load_all_data()
    connectomes = compute_fc_parcel(brain_data)
    write_connectomes(connectomes)
