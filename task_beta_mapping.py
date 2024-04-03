"""
This script will read in task data and run trial-wise first level GLM to extract trial-specific beta maps.
It then masks this data to save a flattened numpy array with the beta map for each trial.
These maps are then loaded in task_decoding.py to decode task states.

Author: Chris Iyer
Updated: 4/2/24
"""

import glob
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
    Retrieves data filenames, event files, and trimmed confounds dataframes given a task
    """
    bids_dir = '/oak/stanford/groups/russpold/data/network_grant/discovery_BIDS_21.0.1/derivatives/glm_data_MNI'
    data_files = [f for f in glob.glob(bids_dir + f'/**/*{task}*optcom_bold.nii.gz', recursive=True) if 'ses-11' not in f and 'ses-12' not in f] # previously optcomDenoised
    confound_files = [f for f in glob.glob(bids_dir + f'/**/*{task}*confounds*', recursive=True) if 'ses-11' not in f and 'ses-12' not in f]

    def process_confound_files(data_files, confound_files): # make sure they correspond to data files, extract to pd df
        confound_dfs = []
        for d in data_files: # need to index on the data because there are some wonky ones with a confound file but no data file
            sub_ses = d[d.find('sub'):d.find('sub')+14]
            c_df = pd.read_csv([f for f in confound_files if sub_ses in f][0], sep='\t')
            c_df = c_df[[col for col in c_df.columns if 'cosine' in col or 'trans' in col or 'rot' in col]] # just get cosine and 24 motion regressors
            confound_dfs.append(c_df)
        return confound_dfs
    confounds = process_confound_files(data_files, confound_files)

    events = [pd.read_csv(f, sep='\t') for f in glob.glob(bids_dir + f'/**/*{task}*events*', recursive=True) if 'ses-11' not in f and 'ses-12' not in f] 
    subjects = [e[e.find('sub') : e.find('sub')+7] for e in data_files]

    return data_files, events, confounds, subjects


def glm_lsa(data_files, events, confounds, subjects, glm_params):
    """Trial-wise GLM modeling, using the Least Squares - All approach.
    Returns:
        - a list (of length n_sessions), where each item is itself a list (of length n_trials) of beta maps. 
        - a list (n_sessions) of lists (n_trials) of label tuples of the format (trial_type, 'correct/incorrect')
    """
    beta_maps = []
    labels = []

    for i_sub, sub in enumerate(subjects):
        sub_beta_maps = []
        sub_labels = []
        glm_params['subject_label'] = sub

        # construct new events df with trial-specific labels
        lsa_events_df = events[i_sub].copy()
        conditions = lsa_events_df['trial_type'].unique()
        condition_counter = {c: 0 for c in conditions}
        for i_trial, trial in lsa_events_df.iterrows():
            trial_condition = trial['trial_type']
            condition_counter[trial_condition] += 1
            lsa_events_df.loc[i_trial, 'trial_type'] = f"{trial_condition}__{condition_counter[trial_condition]:03d}" # new trial-specific label '__'
        
        # fit glm with new events df
        lsa_glm = FirstLevelModel(**glm_params)
        lsa_glm.fit(data_files[i_sub], events=lsa_events_df[['onset','duration','trial_type']], confounds=confounds[i_sub])
        
        # extract beta maps
        for i_trial,trial in lsa_events_df.iterrows():
            beta_map = lsa_glm.compute_contrast(trial['trial_type'], output_type='effect_size') 
            sub_beta_maps.append(beta_map)
            sub_labels.append( ( trial['trial_type'].split('__')[0], trial['correct_response'] == trial['key_press'] ) ) # original trial type and trial outcome

        beta_maps.append(concat_imgs(sub_beta_maps))
        labels.append(sub_labels)

    return beta_maps, labels


def glm_lss(data_files, events, confounds, subjects, glm_params):
    """ NOTE: NOT FUNCTIONAL YET (or efficient enough to actually run? idk)
    Trial-wise GLM modeling, using the Least Squares - Separate approach.
    Returns:
        - a list (of length n_sessions), where each item is itself a list (of length n_trials) of beta maps. 
        - a list of label tuples of the format (trial_type, 'correct/incorrect')
    """
    beta_maps = []
    labels = []

    def label_one_row(df, row_number):
        """Label one trial for one LSS model. Takes events file and row number of trial to model."""
        df = df.copy()

        # Determine which number trial it is *within its condition*
        trial_condition = df.loc[row_number, "trial_type"]
        trials_of_this_type_indices = df["trial_type"].loc[df["trial_type"] == trial_condition].index.tolist()
        trial_number = trials_of_this_type_indices.index(row_number)

        trial_name = f"{trial_condition}__{trial_number:03d}" # new trial-specific label
        df.loc[row_number, "trial_type"] = trial_name
        return df, trial_name
    
    def one_sub(i_sub):
        sub = subjects[i_sub]
        sub_beta_maps = []
        sub_labels = []
        glm_params['subject_label'] = sub

        for i_trial in range(len(events[i_sub])):
            lss_events_df, trial_condition = label_one_row(events[i_sub], i_trial)

            lss_glm = FirstLevelModel(**glm_params)
            lss_glm.fit(data_files[i_sub], events=lss_events_df[['onset','duration','trial_type']], confounds=confounds[i_sub])
            beta_map = lss_glm.compute_contrast(trial_condition, output_type="effect_size")

            sub_beta_maps.append(beta_map)
            sub_labels.append( (trial_condition.split('__')[0], events[i_sub]['correct_response'] == events[i_sub]['key_press']) )# original trial name + trial outcome

        return concat_imgs(sub_beta_maps), sub_labels

    out = Parallel(n_jobs = min(len(subjects), 32) ) (
        delayed(one_sub)(i_sub) for i_sub in range(len(subjects))
    )

    beta_maps = [s[0] for s in out]
    labels = [s[1] for s in out]
    return beta_maps, labels


def extract_beta_series(data_files, events, confounds, subjects, method='LSA'):
    """
    For a given task, this function will run first level GLM to extract trial-wise beta timeseries (LSS or LSA)
        for each subject/session. It will then vectorize the beta maps with nilearn maskers.
    Returns:
        data: a list (of length n_sessions) of beta series data from each trial
        labels: a list (of length n_sessions) of lists (of n_trials) of tuples: (trial_type, <correct: True or False>)
    """
    glm_params = { # SHOULD PULL THESE SPECIFICALLY FROM SOMEWHERE?
        't_r': 1.49,
        'mask_img': get_combined_mask(),
        'noise_model': 'ar1', #???
        'standardize': False,
        'drift_model': None,
        'smoothing_fwhm': 5, #????
        'n_jobs': 32
    }

    if method=='LSA':
        beta_maps, labels = glm_lsa(data_files, events, confounds, subjects, glm_params)
    elif method=='LSS':
        beta_maps, labels = glm_lss(data_files, events, confounds, subjects, glm_params)

    data = MultiNiftiMasker(
        mask_img = get_combined_mask(), # mask where gray matter above 50% and the parcellation applies
        standardize = 'zscore_sample',
        n_jobs = 32
    ).fit_transform(beta_maps)

    return data, labels


if __name__ == "__main__":
    tasks = ['stopSignal','nBack','directedForgetting','goNogo','shapeMatching','spatialTS','cuedTS','flanker']
    for task in tasks:
        data_files, events, confounds, subjects = load_files(task)
        data, labels = extract_beta_series(data_files, events, confounds, subjects, method='LSA')
        np.save(f'/scratch/users/csiyer/glm_outputs/{task}_beta_maps.npy', data)
        np.save(f'/scratch/users/csiyer/glm_outputs/{task}_labels.npy', labels)
        np.save(f'/scratch/users/csiyer/glm_outputs/{task}_subjects.npy', subjects)
        print(f'completed {task}')
        del data, data_files, events, confounds, subjects, labels