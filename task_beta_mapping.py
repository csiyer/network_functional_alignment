"""
This script will read in task data and run trial-wise first level GLM to extract trial-specific beta maps.
It then masks this data to save a flattened numpy array with the beta map for each trial.
These maps are then loaded in task_decoding.py to decode task states.

Author: Chris Iyer
Updated: 4/2/24
"""

import os, glob, pickle
import numpy as np
import pandas as pd
from nilearn.maskers import MultiNiftiMasker
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import concat_imgs
from joblib import Parallel, delayed
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


def glm_lss(sub, d_file, events, confounds, glm_params):
    """ NOTE: NOT FUNCTIONAL YET (or efficient enough to actually run? idk)
    Trial-wise GLM modeling, using the Least Squares - Separate approach.
    Returns:
        - a list (of length n_trials) of beta maps. 
        - a list of label tuples of the format (trial_type, 'correct/incorrect')
    """
    beta_series = []
    label_series = []
    glm_params['subject_label'] = sub
    glm_params['n_jobs'] = 1 # parallelizing on our own

    def relabel_one_row(df, row_number):
        """Label one trial for one LSS model. Takes events file and row number of trial to model."""
        df = df.copy()
        # Determine which number trial it is *within its condition*
        trial_condition = df.loc[row_number, "trial_type"]
        trials_of_this_type_indices = df["trial_type"].loc[df["trial_type"] == trial_condition].index.tolist()
        trial_number = trials_of_this_type_indices.index(row_number)
        trial_name = f"{trial_condition}__{trial_number:03d}" # make new trial-specific label
        df.loc[row_number, "trial_type"] = trial_name
        return df, trial_name
    
    def glm_one_trial(i_trial):
        lss_events_df, trial_condition = relabel_one_row(events, i_trial)
        lss_glm = FirstLevelModel(**glm_params)
        lss_glm.fit(d_file, events=lss_events_df[['onset','duration','trial_type']], confounds=confounds)
        beta_map = lss_glm.compute_contrast(trial_condition, output_type="effect_size")
        
        trial_type = trial_condition.split('__')[0]
        correct = events.correct_response.iloc[i_trial] == events.key_press.iloc[i_trial] 
        return beta_map, (trial_type, correct)

    out = Parallel(n_jobs=32) (
        delayed(glm_one_trial)(i_trial) for i_trial in range(len(events)) 
    )

    beta_series = [s[0] for s in out]
    label_series = [s[1] for s in out]

    return concat_imgs(beta_series), label_series


def extract_save_beta_maps(task, method='LSA'):
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
        subject_session = d_file[d_file.find('sub'):d_file.find('sub')+7] + '_' + d_file[d_file.find('ses'):d_file.find('ses')+6] 
        events = pd.read_csv(e_file, sep='\t')
        confounds = pd.read_csv(c_file, sep='\t')
        confounds = confounds[[col for col in confounds.columns if 'cosine' in col or 'trans' in col or 'rot' in col]] # just get cosine and 24 motion regressors

        if method=='LSA':
            session_beta_maps, session_labels =  glm_lsa(sub, d_file, events, confounds, glm_params)
        elif method=='LSS':
            session_beta_maps, session_labels =  glm_lss(sub, d_file, events, confounds, glm_params)

        session_beta_maps_masked = MultiNiftiMasker(
            mask_img = get_combined_mask(), # mask where gray matter above 50% and the parcellation applies
            standardize = 'zscore_sample',
            n_jobs = 32
        ).fit_transform(session_beta_maps)

        np.save(f'/scratch/users/csiyer/glm_outputs/{task}_{subject_session}_beta_maps.npy', session_beta_maps_masked)
        np.save(f'/scratch/users/csiyer/glm_outputs/{task}_{subject_session}_labels.npy', session_labels)
        
        del session_beta_maps, session_beta_maps_masked, session_labels


def aggregate_saved_maps(task):
    """Searches through saved beta maps + labels files and combines within each task."""
    
    GLM_PATH = f'/scratch/users/csiyer/glm_outputs/{task}'
    task_beta_files = glob.glob(GLM_PATH + '*beta*')
    task_beta_maps = [np.load(f) for f in task_beta_files]
    with open(f'/scratch/users/csiyer/glm_outputs/{task}_beta_maps.pkl', 'wb') as f:
            pickle.dump(task_beta_maps, f)
    f.close()

    task_labels_files = glob.glob(GLM_PATH + '*labels*')
    task_labels = [np.load(f) for f in task_labels_files]
    with open(f'/scratch/users/csiyer/glm_outputs/{task}_labels.pkl', 'wb') as f:
            pickle.dump(task_labels, f)
    f.close()

    for b,l in zip(task_beta_files, task_labels_files): # now delete old files
        os.remove(b)
        os.remove(l)


if __name__ == "__main__":
    tasks = ['stopSignal','nBack','directedForgetting','goNogo','shapeMatching','spatialTS','cuedTS','flanker']
    for task in tasks:
        extract_save_beta_maps(task, method='LSA')
        aggregate_saved_maps(task)
        print(f'completed {task}')