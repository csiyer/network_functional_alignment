"""
This script will read in connectomes created in connectivity.py and tested in reliability.ipynb
and derive (connectivity) Shared Response Models (Chen et al 2015, Guntupalli et al. 2018, Nastase et al. 2019).

Shared Response Modelling derives a single shared space, as well as transformation matrices from 
each subject's native anatomical space into the shared space. These transformation matrices contain
information about subject-specific idiosyncratic functional topographics in the cortex, and these are
what we will later apply to task data to see if they generalize to decoidng cognitive control states. 

We parcellate the cortex and derive one SRM per parcel--this imposes an antomical constraint
on the spatial degree to which voxels can affect shared responses. To get all the subjects'
transformation matrices, we concatenate the SRMs for each parcel to create one whole-brain
transformation matrix per subject, which is saved and used in task_decoding.py

NOTE: the SRM derivation code is copied from the BrianIAK library (https://brainiak.org/)

Author: Chris Iyer
Updated: 9/22/23
"""

import glob
import numpy as np
import nibabel as nib
from nilearn.image import math_img
from connectivity import get_gm_mask, get_parcellation
from utils import srm_brainiak as srm
from joblib import Parallel, delayed

def load_avg_connectomes():
    """load subject-average connectomes (voxels x parcels) computed in reliability.ipynb"""
    sub_list = np.unique([f[f.find('sub'):f.find('sub')+7] for f in glob.glob('/scratch/users/csiyer/*avg*')])
    data_list = [np.load(glob.glob(f'/scratch/users/csiyer/{sub}_connectome_avg.npy')[0]) for sub in sub_list]
    return data_list, sub_list

def load_parcel_map(n_dimensions, atlas='schaefer'):
    """Schaefer 2018 parcel map that corresponds exactly the voxel dimension of the connectomes (Saved in 1_connectivity.py)"""
    gm_mask = get_gm_mask()
    _, parcel_map, parcel_mask = get_parcellation(atlas = atlas, n_dimensions = n_dimensions, resample_target = nib.load('/oak/stanford/groups/russpold/data/network_grant/discovery_BIDS_21.0.1/derivatives/glm_data_MNI/sub-s03/ses-01/func/sub-s03_ses-01_task-rest_run-1_space-MNI_desc-optcomDenoised_bold.nii.gz')) 
    combined_mask = math_img('img1 * img2', img1=gm_mask, img2=parcel_mask)
    parcel_map_flat = parcel_map.get_fdata()[combined_mask.get_fdata() > 0].flatten() # apply same mask we used on the data and flatten
    return parcel_map_flat

def compute_srms(data_list, sub_list, parcel_map, n_features=50, n_iter=20, save=False):
    """
    This function uses BrainIAK's Shared Response Modeling function to compute parcel-wise SRMs (one per parcel, as an anatomical constraint).
    We employ joblib's Parallel and delayed functions to speed up the process.
    
    Inputs:
        - data_list: each element is one subject's average connectome to derive the SRM from
        - sub_list: list of strings of subject identifiers
        - n_features: features in the shared model (default 50, implement grid search later?)
        - n_iter: iterations for SRM. 20 is generally enough to converge
        - save: save the outputs to 'outputs/srm'

    Outputs:
        - subject_transforms: each element is one subject's transformation from voxel space to shared space (n_voxels x n_features)
        - parcelwise_shared_responses: each element is one parcel's 'shared response' in the common model. The meaning of this
            is less clear in the case of connectivity SRMs than traditional response SRMs, so we likely will ignore it--but saving it nonetheless.
    """
    def single_parcel_srm(data_list, parcel_map, parcel_label, n_features):
        parcel_idx = np.where(parcel_map == parcel_label)[0]
        data_parcel = [d[parcel_idx] for d in data_list]
        shared_model = srm.SRM(n_iter=20, features=n_features) 
        shared_model.fit(data_parcel)
        return shared_model.s_, shared_model.w_, parcel_idx

    srm_outputs = Parallel(n_jobs=32)(
        delayed(single_parcel_srm)(data_list, parcel_map, parcel_label, n_features) for parcel_label in np.sort(np.unique(parcel_map))
    )

    parcelwise_shared_responses = [s[0] for s in srm_outputs] # concatenate all the parcelwise shared space responses/connectivities

    subject_transforms = [np.zeros((data_list[0].shape[0], n_features)) for i in range(len(data_list))] # empty initalize

    for _, w_, parcel_idx in srm_outputs: # concatenate transforms into subject-wise all-voxel transformation matrices 
        for i,sub in enumerate(subject_transforms):
            sub[parcel_idx,:] = w_[i]
    # i know there's a better way to do that ^ with more linear algebra. urgh

    if save:
        np.save('/scratch/users/csiyer/parcelwise_shared_responses.npy', parcelwise_shared_responses)
        for i,sub in enumerate(subject_transforms):
            np.save(f'/scratch/users/csiyer/{sub_list[i]}_srm_transform.npy', sub)
    
    return subject_transforms, parcelwise_shared_responses 


if __name__ == "__main__":
    print('beginning SRM')
    data_list, sub_list = load_avg_connectomes()
    parcel_map = load_parcel_map(n_dimensions = 100)
    print('loaded data and parcel map')
    subject_transforms, parcelwise_shared_responses = compute_srms(data_list, sub_list, parcel_map, save=True)
    print('completed SRM')
