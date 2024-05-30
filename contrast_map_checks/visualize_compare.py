"""
Takes the contrasts produced in `contrast_map_avg.py` and visualizes them side-by-side with contrasts produced
in a proper GLM analysis, not performed here.

This is designed to be run locally!

Author: Chris Iyer
Updated: 5/30/2024
"""

import glob, os, sys
import numpy as np
from nilearn import plotting, image
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from connectivity import get_combined_mask

###########################################

OUTPATH = 'outputs/glm/visualizations/'

trial_contrast_files = sorted(glob.glob('outputs/glm/contrast_estimates/*stat*'))
true_contrast_files = sorted(glob.glob('outputs/glm/true_contrasts/*t-test*'))
anat_files = sorted(glob.glob('data/mean_t1s/*'))

subjects = ['s03', 's10', 's19', 's29', 's43']
tasks = ['flanker', 'cuedTS', 'directedForgetting', 'spatialTS', 'stopSignal', 'goNogo',]
contrast_key = {
    'cuedTS': 'task switch cost',
    'directedForgetting': 'neg - con',
    'flanker': 'incongruent - congruent',
    # 'nBack': 'twoBack - oneBack',
    'spatialTS': 'task switch cost',
    # 'shapeMatching': 'shapeMatching_contrast-main_vars',
    'goNogo': 'nogo_success - go',
    'stopSignal': 'stop_failure - go'
}

threshold_val = 2
slices = np.linspace(-30, 70, 5)

############################################

def binarize_and_mask(img, threshold_val):
    return image.binarize_img(img, threshold=threshold_val, mask_img=get_combined_mask(local=True))

############################################


for task in tasks:
    print(task)

    f,  axes = plt.subplots(len(subjects), 2, figsize = (20,len(subjects)*5), squeeze=False)
    plt.suptitle(f'{task.title()}: {contrast_key[task]}. Trial aggregates vs. GLM (thresholded)', fontsize=20)

    trial_contrasts = [f for f in trial_contrast_files if task in f in f]
    true_contrasts = [f for f in true_contrast_files if task in f in f]

    for i,(sub, anat_file, trial_contrast, true_contrast) in enumerate(zip(subjects, anat_files, trial_contrasts, true_contrasts)):
        print(sub)
        
        anat_img = nib.load(anat_file)
        trial_map = binarize_and_mask(nib.load(trial_contrast), threshold_val = threshold_val)
        true_map = binarize_and_mask(nib.load(true_contrast), threshold_val = threshold_val)
        
        plotting.plot_stat_map(trial_map, bg_img=anat_img, cmap='cold_hot', axes=axes[i][0], display_mode='z', cut_coords=slices, title=f'{sub}, trial aggregate contrast')
        plotting.plot_stat_map(true_map, bg_img=anat_img, cmap='cold_hot', axes=axes[i][1], display_mode='z', cut_coords=slices, title=f'{sub}, GLM contrast')

        plt.subplots_adjust(hspace=0)
        f.savefig(OUTPATH +f'task-{task}_contrast_visualizations.pdf')
        # plt.show()
        plt.close()
        