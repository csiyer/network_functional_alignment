""" This script calculates functional connectivity matrices for each session/subject from rest data.

Currently, this is implemented with parcels as our connectivity targets, i.e. the connectivity
matrices are (n_voxels x n_parcels).

It is also possible to implement this voxel-to-voxel (chose not to do this because of both
computational demands and potential functional meaninglessness of voxel correlations) and with
searchlight spheres as our connectivity targets (TO-DO: implement this).

These connectomes are then passed to 2_reliability.py to be assessed for within-subject reliability.
If they look good, we will average or concatenate them and then derive SRMs from them in 3_srm.py.

Author: Chris Iyer
Updated: 7/20/2023
"""

import glob
import numpy as np
from nilearn import datasets
from nilearn.maskers import MultiNiftiMasker, MultiNiftiLabelsMasker, NiftiSpheresMasker
from nilearn.connectome import ConnectivityMeasure
from sklearn.covariance import EmpiricalCovariance
from scipy.stats import pearsonr
from datetime import date

def load_data(FILE_PATHS=[], 
            strategy = 'parcel', 
            schaefer_n_rois=400, 
            sphere_radius=8, 
            sphere_spacing=6):
    """This function loads data in 3 different ways:
        1. (NOT EFFICIENT) Strategy = voxel: extracts direct voxel timeseries
        2. Strategy = parcel: extracts parcel timeseries from schaefer 2018 atlas with a given # of ROIs
        3. (NOT IMPLEMENTED) Strategy = searchlight: extracts timeseries of spheres of a given radius and spacing

        NOTE: the masker.fit_transform functions return an array of the shape (n_TRs x n_voxels)
        """

    if FILE_PATHS == []:
        FILE_PATHS = glob.glob('data/rest/*.nii.gz') # all rest data by default

    masker_args = {
        'standardize': 'zscore_sample', # ??
        'n_jobs': -1,
        # add: mask_img from fmriprep brain mask?
        # not doing any: smoothing, detrend, standardize, low_pass, high_pass, t_r
    }

    if strategy == 'voxel':
        masker = MultiNiftiMasker(mask_strategy = 'whole-brain-template', # or gm-template?
                                  **masker_args)

    elif strategy == 'parcel':
        schaefer_atlas = datasets.fetch_atlas_schaefer_2018(n_rois=schaefer_n_rois, 
                                                        yeo_networks=7, 
                                                        resolution_mm=2, # because our data is too
                                                        data_dir='data/schaefer', 
                                                        verbose=0)
        masker = MultiNiftiLabelsMasker(labels_img = schaefer_atlas.maps,
                                labels = schaefer_atlas.labels,
                                resampling_target = 'data',
                                strategy = 'mean',
                                **masker_args)

    elif strategy == 'searchlight':
        sphere_coords = [] # get the centerpoint coordinates of spheres - these current numbers are incorrect
        for x in range(-90, 91, sphere_spacing):
            for y in range(-126, 91, sphere_spacing):
                for z in range(-72, 73, sphere_spacing):
                    sphere_coords.append((x, y, z))
        masker = NiftiSpheresMasker(seeds = sphere_coords, 
                                    radius=sphere_radius, 
                                    **masker_args)
        return [masker.fit_transform(f) for f in FILE_PATHS]

    # this only works for the multimaskers with voxel/parcel
    return masker.fit_transform(FILE_PATHS)

def compute_fc_voxel(voxel_timeseries, cov_estimator=EmpiricalCovariance()):
    """ Full voxel-to-voxel connectivity/correlation matrix
        If passed the parcel timeseries, this will compute a parcel-to-parcel matrix.
        NOTES:
            - This will take forever with the current implementation -- *replace with FCMA toolbox?*
            - Default nilearn covariance estimator is LedoitWolf, replacing here with EmpiricalCovariance() for pearson
    """
    correlation_measure = ConnectivityMeasure(kind="correlation", cov_estimator=cov_estimator)
    return correlation_measure.fit_transform(voxel_timeseries)

def correlate_rows(mat1, mat2):
    """ Helper function for below
        Returns a matrix with the Pearson r correlation of each column (voxel) of mat1 with each column (parcel/target) of mat2
    """
    correlation_matrix = np.empty((mat1.shape[1], mat2.shape[1]))
    for i in range(mat1.shape[0]):
        for j in range(mat2.shape[0]):
            correlation_matrix[i, j] = pearsonr(mat1[:, i], mat2[:, j])[0]
    return correlation_matrix

def compute_fc_target(voxel_timeseries, target_timeseries):
    """ This will take each column in the voxel timeseries (across all the TRs/rows) 
        and correlate it with each column in the target timeseries.
        Connectivity targets could be the parcel timeseries, or a searchlight timeseries
        
        NOTE: The connectivity target in which a given voxel resides is not excluded yet -- should it be?
    """
    return [correlate_rows(voxel_timeseries[i], target_timeseries[i]) for i in range(len(voxel_timeseries))]

# nilearn.plotting.plot_connectome?

def write_connectomes(connectomes):
    # write our outputs back to files to read in for later use
    outpath = 'outputs/connectomes/'
    ids = [file[file.find('sub'): file.find('sub')+14] for file in glob.glob('data/rest/*.nii.gz')]

    for sub in range(len(connectomes)):
        connectomes[sub].tofile(outpath + ids[sub] + '_connectome' + date.today())
    

if __name__ == "__main__":
    brain_data = load_data(strategy='parcel')
    connectomes = compute_fc_target(brain_data)
    write_connectomes(connectomes)
