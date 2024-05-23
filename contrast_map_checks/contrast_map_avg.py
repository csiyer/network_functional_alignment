"""
This is a data-quality sanity check for the trialwise beta maps going into the decoding analysis.
We are concerned that the trial timings are too quick to get solid activity estimates, and that these
beta maps are garbage.

So, here we will model these maps (violating lots of statistical assumptions) with Nilearn SecondLevelModel
to derive subject-level contrast maps that we can compare to contrast maps from our traditional GLM analysis.

If they look at all similar, that is a good sign!

1) First, I re-derive trial-wise beta maps (because the images were not saved before)
2) For each subject, I run SecondLevelModel to derive contrast estimates on those beta maps
3) Compare to GLM-created subject-level fixed effect maps

Author: Chris Iyer
Updated: 5/22/2024
"""

import os,sys
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel

HOME_DIR = '/home/users/csiyer/network_functional_alignment'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(HOME_DIR), '..')))
from connectivity import get_combined_mask
from task_beta_mapping import load_files, replace_trial_types


def edit_session_labels_gng(session_labels):  # change 'no-go' to 'no-go_success' / 'no-go_failure'
    new_session_labels = [
        (l[0] + ('_success' if l[1] else '_failure'), l[1]) if l[0] == 'no-go' else l
            for l in session_labels
    ]
    return new_session_labels


def glm_lsa_edit(sub, d_file, events, confounds, glm_params):
    """   ***Edited to not concatenate images***
    Trial-wise GLM modeling, using the Least Squares - All approach.
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

    if 'no-go' in [l[0] for l in label_series]: label_series = edit_session_labels_gng(label_series) 

    return beta_series, label_series


def extract_beta_imgs(tasks):
    """Step 1: re-derive beta maps and save each trial as its own image"""
    TEMP_PATH = '/scratch/users/csiyer/glm_outputs/trial_beta_imgs/'
    if not os.path.isdir(TEMP_PATH):
        os.mkdir(TEMP_PATH)

    glm_params = { 
        't_r': 1.49,
        'mask_img': get_combined_mask(), # DO I WANT THIS?
        'noise_model': 'ar1', #???
        'standardize': False,
        'drift_model': None,
        'smoothing_fwhm': 5, #????
        'n_jobs': 32
    }

    for task in tasks:
        print('starting first level, task: ' + task)
        subjects, data_files, event_files, confounds_files = load_files(task)
        
        sub_tracker = {}
        for sub in np.unique(subjects):
            sub_tracker[sub] = {'imgs': [], 'labels': []}

        for sub, d_file, e_file, c_file in zip(subjects, data_files, event_files, confounds_files):
            events = pd.read_csv(e_file, sep='\t')
            events = replace_trial_types(events, task)
            confounds = pd.read_csv(c_file, sep='\t')
            confounds = confounds[[col for col in confounds.columns if 'cosine' in col or 'trans' in col or 'rot' in col]] # just get cosine and 24 motion regressors

            session_beta_maps, session_labels =  glm_lsa_edit(sub, d_file, events, confounds, glm_params)
            sub_tracker[sub]['imgs'].extend(session_beta_maps)
            sub_tracker[sub]['labels'].extend(session_labels)

            #subject_session = d_file[d_file.find('sub'):d_file.find('sub')+7] + '_' + d_file[d_file.find('ses'):d_file.find('ses')+6] 
            # for i,(beta,label) in enumerate(zip(session_beta_maps, session_labels)):
            #     trial_label = f'{task}_{subject_session}_trial-{i}_{label[0]}_'
            #     nib.save(beta, TEMP_PATH + trial_label + 'beta_map.nii.gz')
            #     sub_label_tracker[sub].append(label)
            
            del session_beta_maps, session_labels
        
        for sub in np.unique(subjects):
            np.save(TEMP_PATH + f'{task}_{sub}_all-imgs.npy', sub_tracker[sub]['imgs'])
            np.save(TEMP_PATH + f'{task}_{sub}_all-labels.npy', sub_tracker[sub]['labels'])


def recreate_contrasts(tasks):
    """Step 2: for each subject, reconstruct contrasts of interest with the maps created above"""

    task_contrast_key = {
        'cuedTS': {'tswitch_cswitch-tstay_cswitch' },
        'directedForgetting': {'neg-con' },
        'flanker': {'incongruent-congruent' },
        'spatialTS': {'tswitch_cswitch-tstay_cswitch' },
        'goNogo': {'no-go_success-go' }, 
        'stopSignal': {'stop_failure-go' },
    }
    TEMP_PATH = '/scratch/users/csiyer/glm_outputs/trial_beta_imgs/'
    OUTPATH  = '/scratch/users/csiyer/glm_outputs/contrast_estimates/'

    for task in tasks:
        print('starting second level, task: ' + task)
        subjects = np.load('/scratch/users/csiyer/glm_outputs/' + task + '_subjects.npy')
        for sub in np.unique(subjects):
            
            beta_list = np.load(TEMP_PATH + f'{task}_{sub}_all-imgs.npy', allow_pickle=True)
            label_list = np.load(TEMP_PATH + f'{task}_{sub}_all-labels.npy')

            # create design matrix
            df = pd.DataFrame(label_list, columns=['label'])
            desmat = pd.get_dummies(df['label'], dtype=int) # converts to one-hot dummy matrix
            desmat.index = df.index 

            # run secondlevelmodel and extract contrast
            model = SecondLevelModel.fit(beta_list, design_matrix=desmat, n_jobs=8)
            contrast_outputs = model.compute_contrast(task_contrast_key[task], output_type='all')

            # Save each output map
            for map_type, img in contrast_outputs.items():
                nib.save(img, OUTPATH + f'{sub}_{task}_{task_contrast_key[task]}_{map_type}.nii.gz')


if __name__ == '__main__':
    tasks = ['flanker','spatialTS','cuedTS','directedForgetting','stopSignal','goNogo'] # 'shapeMatching',
    # extract_beta_imgs(tasks)
    # print('finished beta images')
    recreate_contrasts(tasks)
