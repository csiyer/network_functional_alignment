import glob
import numpy as np
import nibabel as nib
from nilearn.image import math_img, mean_img
from joblib import Parallel, delayed
from connectivity import get_parcellation
# from nilearn.plotting import view_img, plot_glass_brain

files = glob.glob('/oak/stanford/groups/russpold/data/network_grant/discovery_BIDS_21.0.1/derivatives/glm_data_MNI/**/*rest*optcomDenoised*.nii.gz', recursive = True)
sublist = np.unique([f[f.find('sub'):f.find('sub')+7] for f in files])

def get_missing_voxels(within_sub_missing_allowed=11, mask=''):
    """
    within_sub_missing_allowed = 11: if a voxel has no signal in all 12 of one subject's sessions, exclude it for everyone (most liberal)
    within_sub_missing_allowed = 6: if a voxel has no signal in 6 or more of one subject's sessions, exclude it
    within_sub_missing_allowed = 0: if a voxel has no signal in a single session, exclude it for everyone (most conservative)
    """

    mask = mask.get_fdata()
    master_missing_img = np.zeros((97,115,97)) # keeps track of which voxels have missing values

    for sub in sublist:
        subfiles = [f for f in files if sub in f]
        def one_session(f):
            session_zeros = math_img('img == 0', img = mean_img(nib.load(f))).get_fdata()
            return session_zeros * mask

        sessions = Parallel(n_jobs = -1, backend='threading')(
            delayed(one_session)(f) for f in subfiles
        )
        sub_missing_counts = sum(sessions)
        master_missing_img += (sub_missing_counts > within_sub_missing_allowed).astype(int) 

    master_missing_img = (master_missing_img > 1).astype(int) # back to 1s and 0s

    return master_missing_img

_, _, parcel_mask = get_parcellation(schaefer_n_rois = 400, resample_target = nib.load(files[0]))
gm_mask = math_img('img >= 0.5', img=nib.load('data/templates/tpl-MNI152NLin2009cAsym_res-02_label-GM_probseg.nii.gz'))
double_mask = math_img('img1 * img2', img1 = gm_mask, img2 = parcel_mask) # this will exclude the cerebellum (and probably other voxels)

count = 0
for missing_num in [0,6,11]:
    for mask in ['gm', 'double']:
        if mask == 'gm':
            out = get_missing_voxels(within_sub_missing_allowed = missing_num, mask = gm_mask)
            count += 1
        else: 
            out = get_missing_voxels(within_sub_missing_allowed = missing_num, mask = double_mask)
            count += 1
        print(f'count done: {count}, saving now')
        np.save(f'/scratch/users/csiyer/voxel_test/missing-{missing_num}_mask-{mask}.npy', out)

