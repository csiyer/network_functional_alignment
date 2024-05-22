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

