"""
The decoding analysis is one way to quantify whether SRM seems to be picking up on between-subject signal 
in neural data during these control tasks. Another way to do this is to analyze GLM-derived subject-level 
contrasts directly, rather than trial-level decoding.

Here, we align each subject's GLM-derived contrast maps into shared space using the SRM transforms,
and compare across-subject alignment, in 2 ways:
    1) # of overlapping voxels in thresholded maps (SRM-transformed vs. not)
    2) Dice coefficient of unthresholded maps (SRM-transformed vs. not)

Author: Chris Iyer
Updated: 5/22/2024
"""

import os, sys, glob
import numpy as np
import nibabel as nib

subjects, data_files, event_files, confounds_files = load_files(task)

SRM_DIR = '/scratch/users/csiyer/srm_outputs/'
srm_files = glob.glob(SRM_DIR + '*transform*')

tasks = ['flanker','spatialTS','cuedTS','directedForgetting','stopSignal','goNogo', 'shapeMatching', 'nBack']
for task in tasks:
    subjects = np.load(f'/scratch/users/csiyer/glm_outputs/{task}_subjects.npy') 

    for i,sub in enumerate(subjects):
        srm_transform = np.load([s for s in srm_files if sub in s][0])
        # np.dot(data[i], srm_transform)
    