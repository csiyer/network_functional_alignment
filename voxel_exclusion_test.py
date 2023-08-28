import glob
import numpy as np
import nibabel as nib
from nilearn import datasets
from nilearn.image import math_img
from nilearn.plotting import view_img, plot_glass_brain
from connectivity import get_parcellation

files = glob.glob('/oak/stanford/groups/russpold/data/network_grant/discovery_BIDS_21.0.1/derivatives/glm_data_MNI/**/*rest*optcomDenoised*.nii.gz', recursive = True)
sublist = np.unique([f[f.find('sub'):f.find('sub')+7] for f in files])

def get_missing_voxels(within_sub_missing_allowed=11, mask=''):
    """
    within_sub_missing_allowed = 11: if a voxel has no signal in all 12 of one subject's sessions, exclude it for everyone (most liberal)
    within_sub_missing_allowed = 6: if a voxel has no signal in 6 or more of one subject's sessions, exclude it
    within_sub_missing_allowed = 0: if a voxel has no signal in a single session, exclude it for everyone (most conservative)
    """
    master_missing_img = np.zeros((95,115,97)) # keeps track of which voxels have missing values

    for sub in sublist:
        subfiles = [f for f in files if sub in f]

        missing_count_img = np.zeros((95,115,97))

        for f in subfiles:
            session_data = nib.load(f).get_fdata()
            for x,y,z in np.ndindex(session_data.shape[0:3]):
                if mask[x,y,z] == 1 and np.all(session_data[x,y,z,:] == 0): # the voxel is in the mask and has no signal
                    missing_count_img[x,y,z] += 1
        
        # would this subject's average show a missing voxel here, under this exclusion policy? If so, make our master missing img > 0 to flag it
        master_missing_img += (missing_count_img > within_sub_missing_allowed).astype(int) 

    master_missing_img = (master_missing_img > 1).astype(int) # back to 1s and 0s

    return master_missing_img


_, _, parcel_mask = get_parcellation(schaefer_n_rois = 400, resample_target = nib.load(files[0]))
gm_mask = math_img('img >= 0.5', img=nib.load('data/templates/tpl-MNI152NLin2009cAsym_res-02_label-GM_probseg.nii.gz'))
double_mask = math_img('img1 * img2', img1 = gm_mask, img2 = parcel_mask) # this will exclude the cerebellum (and probably other voxels)

for missing_num in [0,6,11]:
    for mask in ['gm', 'double']:
        if mask == 'gm':
            out = get_missing_voxels(within_sub_missing_allowed = missing_num, mask = gm_mask.get_fdata())
        else: 
            out = get_missing_voxels(within_sub_missing_allowed = missing_num, mask = double_mask.get_fdata())
        np.save(f'/scratch/users/csiyer/voxel_test/missing-{missing_num}_mask-{mask}.npy', out)

