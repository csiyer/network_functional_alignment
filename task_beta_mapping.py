"""
This script will read in task data and run trial-wise first level GLM to extract trial-specific beta maps.
It then masks this data to save a flattened numpy array with the beta map for each trial.
These maps are then loaded in task_decoding.py to decode task states.

Author: Chris Iyer
Updated: 5/2/24
"""

import os, glob, pickle
import numpy as np
import pandas as pd
from nilearn.maskers import MultiNiftiMasker
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import concat_imgs
from connectivity import get_combined_mask
# /oak/stanford/groups/russpold/data/network_grant/discovery_BIDS_21.0.1/derivatives/glm_data_MNI

def load_files(task):
    """
    Retrieves filenames of data, events, confounds. Saves list of subjects for later use. 
    """
    bids_dir = '/oak/stanford/groups/russpold/data/network_grant/discovery_BIDS_21.0.1/derivatives/glm_data_MNI'
    data_files = [f for f in glob.glob(bids_dir + f'/**/*{task}*optcom_bold.nii.gz', recursive=True) if 'ses-11' not in f and 'ses-12' not in f] # previously optcomDenoised
    subjects = [e[e.find('sub') : e.find('sub')+7] for e in data_files]
    event_files = [f for f in glob.glob(bids_dir + f'/**/*{task}*events*', recursive=True) if 'ses-11' not in f and 'ses-12' not in f] 
    np.save(f'/scratch/users/csiyer/glm_outputs/{task}_subjects.npy', subjects)
    confounds_files = [f for f in glob.glob(bids_dir + f'/**/*{task}*confounds*', recursive=True) if 'ses-11' not in f and 'ses-12' not in f]

    # a couple erroneous / unmatched confounds files to exclude
    all_subjects_sessions = [d[d.find('sub'):d.find('sub')+14] for d in data_files]
    confounds_files = [c for c in confounds_files if c[c.find('sub'):c.find('sub')+14] in all_subjects_sessions]

    return subjects, data_files, event_files, confounds_files


def replace_trial_types(events_df, task):
    """Replaces values in the trial_type column with desired trial types for decoding, excludes NAs"""
    
    with open('utils/task_decoding_conditions.json', 'r') as file:
        task_conditions = eval(file.read())
    file.close()

    column_of_interest = task_conditions[task]['colname']
    accepted_values = [i for i in task_conditions[task]['values'].keys()]

    new_df = events_df.copy()
    new_df = new_df[ new_df[column_of_interest].isin(accepted_values)] # exclude NA trials and others
    new_df.trial_type = new_df[column_of_interest].map(task_conditions[task]['values']) # replace with desired value as necessary
    
    return new_df


def glm_lsa(sub, d_file, events, confounds, glm_params):
    """Trial-wise GLM modeling, using the Least Squares - All approach.
    Returns:
        - a list (of length n_trials) of beta maps (nifti objects) 
        - a list (n_trials) of label tuples of the format (trial_type, 'correct/incorrect')
    """
    beta_series = []
    label_series = []
    glm_params['subject_label'] = sub

    # construct new events df with trial-specific labels
    lsa_events_df = events.copy()
    conditions = lsa_events_df['trial_type'].unique()
    condition_counter = {c: 0 for c in conditions}
    for i_trial, trial in lsa_events_df.iterrows():
        trial_condition = trial['trial_type']
        condition_counter[trial_condition] += 1
        lsa_events_df.loc[i_trial, 'trial_type'] = f"{trial_condition}__{condition_counter[trial_condition]:03d}" # new trial-specific label with '__'
    
    # fit glm with new events df
    lsa_glm = FirstLevelModel(**glm_params)
    lsa_glm.fit(d_file, events=lsa_events_df[['onset','duration','trial_type']], confounds=confounds)
    
    # extract beta maps
    for i_trial,trial in lsa_events_df.iterrows():
        beta_map = lsa_glm.compute_contrast(trial['trial_type'], output_type='effect_size') 
        beta_series.append(beta_map)

        trial_type = trial['trial_type'].split('__')[0]
        correct = trial['correct_response'] == trial['key_press']
        label_series.append( (trial_type, correct) ) 

    return concat_imgs(beta_series), label_series


def extract_save_beta_maps(task):
    """
    For a given task, this function will load necessary files and run
    first level GLM to extract trial-wise beta timeseries (LSS or LSA)
    for each subject/session. It will then vectorize the maps with nilearn maskers and save as .npy files. 
    """
    subjects, data_files, event_files, confounds_files = load_files(task)

    glm_params = { # SHOULD PULL THESE SPECIFICALLY FROM SOMEWHERE?
        't_r': 1.49,
        'mask_img': get_combined_mask(),
        'noise_model': 'ar1', #???
        'standardize': False,
        'drift_model': None,
        'smoothing_fwhm': 5, #????
        'n_jobs': 32
    }

    for sub, d_file, e_file, c_file in zip(subjects, data_files, event_files, confounds_files):
        events = pd.read_csv(e_file, sep='\t')
        events = replace_trial_types(events, task)

        confounds = pd.read_csv(c_file, sep='\t')
        confounds = confounds[[col for col in confounds.columns if 'cosine' in col or 'trans' in col or 'rot' in col]] # just get cosine and 24 motion regressors

        session_beta_maps, session_labels =  glm_lsa(sub, d_file, events, confounds, glm_params)

        session_beta_maps_masked = MultiNiftiMasker(
            mask_img = get_combined_mask(), # mask where gray matter above 50% and the parcellation applies
            standardize = 'zscore_sample',
            n_jobs = 32
        ).fit_transform(session_beta_maps)

        subject_session = d_file[d_file.find('sub'):d_file.find('sub')+7] + '_' + d_file[d_file.find('ses'):d_file.find('ses')+6] 
        np.save(f'/scratch/users/csiyer/glm_outputs/{task}_{subject_session}_beta_maps.npy', session_beta_maps_masked)
        np.save(f'/scratch/users/csiyer/glm_outputs/{task}_{subject_session}_labels.npy', session_labels)
        
        del session_beta_maps, session_beta_maps_masked, session_labels


def aggregate_saved_maps(task):
    """Searches through saved beta maps + labels files and combines within each task."""
    
    GLM_PATH = f'/scratch/users/csiyer/glm_outputs/{task}'
    task_beta_files = sorted(glob.glob(GLM_PATH + '*sub*beta*'))
    task_beta_maps = [np.load(f) for f in task_beta_files]
    with open(f'/scratch/users/csiyer/glm_outputs/{task}_beta_maps.pkl', 'wb') as f:
            pickle.dump(task_beta_maps, f)
    f.close()

    task_labels_files = sorted(glob.glob(GLM_PATH + '*sub*labels*'))
    task_labels = [np.load(f) for f in task_labels_files]
    with open(f'/scratch/users/csiyer/glm_outputs/{task}_labels.pkl', 'wb') as f:
            pickle.dump(task_labels, f)
    f.close()

    for b,l in zip(task_beta_files, task_labels_files): # now delete old files
        os.remove(b)
        os.remove(l)


if __name__ == "__main__":
    tasks = ['flanker','spatialTS','cuedTS','shapeMatching','directedForgetting','stopSignal','goNogo']
    for task in tasks:
        extract_save_beta_maps(task)
        aggregate_saved_maps(task)
        print(f'completed {task}')