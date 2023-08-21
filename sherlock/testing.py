
import glob
import numpy as np
import nibabel as nib
from nilearn import datasets
from nilearn.image import resample_img, math_img
from nilearn.maskers import MultiNiftiMasker, MultiNiftiLabelsMasker

BIDS_DIR = '/oak/stanford/groups/russpold/data/network_grant/discovery_BIDS_21.0.1/derivatives/glm_data_MNI'

files = glob.glob(BIDS_DIR + '/**/*rest*optcomDenoised*.nii.gz', recursive = True)
confounds_files = glob.glob(BIDS_DIR + '/**/*rest*confounds*.tsv', recursive = True)

FILE_PATHS = files[0:3]
CONFOUNDS_FILES = confounds_files[0:3]

subject_session_list = [(f[f.find('sub'):f.find('sub')+7], f[f.find('ses'):f.find('ses')+6]) for f in FILE_PATHS]

schaefer_atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400, 
                                                        yeo_networks=7, 
                                                        resolution_mm=2,
                                                        data_dir='data/templates', 
                                                        verbose=0)
schaefer_resampled = resample_img(schaefer_atlas.maps, 
                                    target_shape = nib.load(FILE_PATHS[0]).shape[:3],
                                    target_affine = nib.load(FILE_PATHS[0]).affine,
                                    interpolation = 'nearest')


parcel_labels = schaefer_atlas.labels
parcel_map = schaefer_resampled
parcel_mask = math_img('img > 0', img=schaefer_resampled)

parcel_map_flat = parcel_map.get_fdata()[parcel_map.get_fdata() > 0].flatten()  

masker_args = {
    'standardize': 'zscore_sample', # ?
    'n_jobs': -1,
}

# parcel_masker = MultiNiftiLabelsMasker(
#     labels_img = parcel_map,
#     labels = parcel_labels,
#     **masker_args
# )
# parcel_data = parcel_masker.fit_transform(FILE_PATHS, confounds = CONFOUNDS_FILES)


voxel_masker = MultiNiftiMasker(
    mask_img = parcel_mask, # CRUCIAL: need this to be the case so we can know which parcel each voxel belongs to later
    **masker_args
)

voxel_data = voxel_masker.fit_transform(FILE_PATHS)
voxel_data = voxel_masker.fit_transform(FILE_PATHS, confounds = CONFOUNDS_FILES)