"""
1) For each subject, for each task condition, average all beta maps across trials/sessions. 
2) For each subject, reconstruct contrasts of interest with the maps created above
3) Compare to GLM-created subject-level fixed effect maps

Author: Chris Iyer
Updated: 5/22/2024
"""

import os, pickle, glob
import numpy as np

def average_and_save_maps():
    """Step 1: for each subject, for each task condition, average all beta maps across trials/sessions"""
    
    INPATH = '/scratch/users/csiyer/glm_outputs/'
    OUTPATH = '/scratch/users/csiyer/glm_outputs/sub_avg/'
    if not os.path.isdir(OUTPATH):
        os.mkdir(OUTPATH)

    def edit_gng_labels(labels_all):  # change 'no-go' to 'no-go_success' / 'no-go_failure'
        new_labels = []
        for session_labels in labels_all:
            new_session_labels = [
                (l[0] + ('_success' if l[1] else '_failure'), l[1]) if l[0] == 'no-go' else l
                for l in session_labels
            ]
            new_labels.append(new_session_labels)
        return new_labels

    for task in ['flanker','spatialTS','cuedTS','directedForgetting','stopSignal','goNogo']: # 'shapeMatching', 'nBack'
        subjects = np.load(INPATH + task + '_subjects.npy')
        with open(f'/scratch/users/csiyer/glm_outputs/{task}_beta_maps.pkl', 'rb') as f:
                data = pickle.load(f)
        f.close()
        with open(f'/scratch/users/csiyer/glm_outputs/{task}_labels.pkl', 'rb') as f:
                labels_all = pickle.load(f) # each item is a list of tuples of format (condition, correct_boolean)
        f.close()

        if task == 'goNogo': labels_all = edit_gng_labels(labels_all)
        
        for sub in subjects:
            sub_idxs = [i for i in range(len(subjects)) if subjects[i] == sub]

            for label in np.unique([label for label,correct in labels_all[sub_idxs[0]]]):
                maps_to_avg = []
                # for each unique label, loop through sessions and add all the maps from trials of that label
                for sess_i in sub_idxs: 
                    maps_to_add = [d for d,l in zip(data[sess_i], labels_all[sess_i]) if l[0] == label]
                    maps_to_avg.extend(maps_to_add)

                sub_label_avg = np.mean(maps_to_avg, axis=0)
                np.save(OUTPATH + f'{sub}_{task}_{label}_avg_beta_map.npy', sub_label_avg)
    return 'done!'


def recreate_contrasts():
    """Step 2: for each subject, reconstruct contrasts of interest with the maps created above"""

    contrasts_of_interest = {
        'cuedTS': 'cuedTS_contrast-task_switch_cost',
        'directedForgetting': 'directedForgetting_contrast-neg-con',
        'flanker': 'flanker_contrast-incongruent - congruent',
        # 'nBack': 'nBack_contrast-twoBack-oneBack',
        'spatialTS': 'spatialTS_contrast-task_switch_cost',
        # 'shapeMatching': 'shapeMatching_contrast-main_vars',
        'goNogo': 'goNogo_contrast-nogo_success-go',
        'stopSignal': 'stopSignal_contrast-stop_failure-go'
    }
    task_contrast_key = {
        'cuedTS': {'first': 'tswitch_cswitch', 'minus': 'tstay_cswitch' },
        'directedForgetting': {'first': 'neg', 'minus': 'con' },
        'flanker': {'first': 'incongruent', 'minus': 'congruent' },
        # 'nBack': {'first': '', 'minus': '' },
        'spatialTS': {'first': 'tswitch_cswitch', 'minus': 'tstay_cswitch' },
        # 'shapeMatching': {'first': '', 'minus': '' },
        'goNogo': {'first': 'no-go_success', 'minus': 'go' }, 
        'stopSignal': {'first': 'stop_failure', 'minus': 'go' },
    }
    INPATH = '/scratch/users/csiyer/glm_outputs/sub_avg/'
    OUTPATH = '/scratch/users/csiyer/glm_outputs/sub_avg_contrasts/'
    if not os.path.isdir(OUTPATH):
        os.mkdir(OUTPATH)

    for task in ['flanker','spatialTS','cuedTS','directedForgetting','stopSignal','goNogo']: # 'shapeMatching', 'nBack'
        subjects = np.load(INPATH + task + '_subjects.npy')
        for sub in np.unique(subjects):
            first_condition_file = INPATH + f'{sub}_{task}_{task_contrast_key[task]["first"]}_avg_beta_map.npy'
            minus_condition_file = INPATH + f'{sub}_{task}_{task_contrast_key[task]["minus"]}_avg_beta_map.npy'

            contrast_map = np.load(first_condition_file) - np.load(minus_condition_file)
            np.save(OUTPATH + f'{sub}_{task}_{contrasts_of_interest[task]}_estimate.npy', contrast_map)

    return 'done!'
    

if __name__ == "__main__":
    average_and_save_maps()
    recreate_contrasts()