"""
Here, we use transformation matrices from srm.py to transform task data into the shared space.

Then, we decode trial conditions (e.g., congruent vs. incongruent; see full condition list below) 
between subjects using leave-one-subject-out cross-validation, with and without the SRM transformation. 
In each fold, all of one subject's sessions are excluded from the training set and tested on, using classifiers trained
on either SRM'd or raw data from all other subjects. Each TR is classified according to its trial condition.

Assessing the performance benefit of SRM transformation tests how functionally shared or idiosyncratic
neural signatures of these cognitive control tasks are.

NEXT STEPS:
    - Follow this tutorial: https://nilearn.github.io/dev/auto_examples/02_decoding/plot_haxby_glm_decoding.html#sphx-glr-auto-examples-02-decoding-plot-haxby-glm-decoding-py
        in order to decode trial-level beta maps instead of raw data?
    - Differentiate trial types based on behavioral outcome (correct/incorrect response)

Author: Chris Iyer
Updated: 12/11/23
"""

import glob, json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.svm import LinearSVC
import nibabel as nib
from nilearn.maskers import MultiNiftiMasker
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from connectivity import get_combined_mask
# /oak/stanford/groups/russpold/data/network_grant/discovery_BIDS_21.0.1/derivatives/glm_data_MNI

def load_data(task):
    """
    Loads all data for a single task (5 subjects x 5 sessions each = 25 files).
    Returns: 
        - list of data from MultiNiftiMasker
        - list of event files in the same order
        - subject list matching the two above^
    """
    bids_dir = '/oak/stanford/groups/russpold/data/network_grant/discovery_BIDS_21.0.1/derivatives/glm_data_MNI'

    data_files = [f for f in glob.glob(bids_dir + f'/**/*{task}*optcomDenoised_bold.nii.gz', recursive=True) if 'ses-11' not in f and 'ses-12' not in f]
    # confound_files = glob.glob(bids_dir + f'/**/*{task}*confounds*', recursive=True)
    event_files = [f for f in glob.glob(bids_dir + f'/**/*{task}*events*', recursive=True) if 'ses-11' not in f and 'ses-12' not in f] 
    
    data = MultiNiftiMasker(
        mask_img = get_combined_mask(), # mask where it's gray matter above 50% and the parcellation applies
        standardize = 'zscore_sample',
        n_jobs = 32
    ).fit_transform(data_files) # , confounds = confound_files)

    events = [pd.read_csv(e, sep='\t') for e in event_files]
    subjects = [e[e.find('sub') : e.find('sub')+7] for e in event_files]

    return data, events, subjects


def average_trials(data, events):
    """
        (1) Average the trials within each trial, accounting for a 4.5s HRF lag and a 1.49s TR.
        (2) eliminate images/labels from 'NA" trials
    """
    if len(data) != len(events):
        return "ERROR: number of sessions do not match"
    
    hrf_lag = 4.5
    def time_to_tr(time): # for a given point in time, what TR contains activity relating to that time, accounting for HRF lag and 1.49s TR?
        return (time+hrf_lag) / 1.49
    
    data_trialaveraged = []
    labels = []
    
    for i_ses,(d,e) in enumerate(zip(data,events)): # for each subject/session

        data_trialaveraged.append(np.zeros((len(e), d.shape[1]))) # new data matrix, n_trials x n_voxels
        labels.append([]) 

        for j_trial in range(len(e)): # for each trial
            start_time = e.onset.iloc[j_trial]
            if j_trial == len(e) - 1: 
                stop_time = start_time + 6 # somewhat arbitrary stop for final trial
            else:
                stop_time = e.onset.iloc[j_trial+1]
            
            start_tr = int(np.floor(time_to_tr(start_time)))
            stop_tr = int(np.floor(time_to_tr(stop_time)))

            data_trialaveraged[i_ses][j_trial,] = np.mean(d[start_tr:stop_tr,], axis=0)
            labels[i_ses].append(e.trial_type.iloc[j_trial])

        # lastly, eliminate NA trials for this session
        not_na = [i for i,l in enumerate(labels[i_ses]) if l != 'na']
        data_trialaveraged[i_ses] = np.array(data_trialaveraged[i_ses])[not_na]
        labels[i_ses] = np.array(labels[i_ses])[not_na]

    return data_trialaveraged, labels


def label_trs(data, events, task, correct_only=False):
    """
    Instead of averaging TRs within each trial, this assigns each TR a trial label.
    Uses task_decoding_condition.json to define the trials that we keep / label. (eliminates NAs and irrelevant trials)
    correct_only filters for only trials with correct responses.
    """
    with open('utils/task_decoding_conditions.json', 'r') as file:
        task_conditions = eval(file.read())

    hrf_lag = 4.5
    def tr_to_time(tr): # for the Nth tr, from which timepoint does this TR contain brain information?
        return tr*1.49 - hrf_lag
    
    def find_active_trial(tr,e): # for a given time, what was the active trial?
        time = tr_to_time(tr)
        if time < e.onset.iloc[0] or time > e.onset.iloc[-1] + 6:
            return None
        else:
            trial_num = e.onset[e.onset <= time].idxmax()
            trial_type = e[task_conditions[task]['colname']].iloc[trial_num]
            correct = e.correct_response.iloc[trial_num] == e.key_press.iloc[trial_num]
            if (correct_only and not correct) or (trial_type not in task_conditions[task]['values']): # either incorrect trial or excluded trial type
                return None
            return task_conditions[task]['values'][trial_type]
        
    labels = [np.array([find_active_trial(i,e) for i in range(d.shape[0])]) for d, e in zip(data, events)]
    
    # remove NA/excluded trials
    data_trimmed = [d[np.where(l != None)] for d,l in zip(data, labels)] 
    labels = [l[np.where(l != None)] for l in labels]

    return data_trimmed, labels


def srm_transform(data, subjects, zscore=True):
    data_srm = []
    srm_dir = '/scratch/users/csiyer/srm_outputs/'
    srm_files = glob.glob(srm_dir + '*transform*')
    for i,sub in enumerate(subjects):
        srm_transform = np.load([s for s in srm_files if sub in s][0])
        curr = np.dot(data[i], srm_transform)
        if zscore:
            curr = StandardScaler().fit_transform(curr)
        data_srm.append(curr)
    return data_srm


def loso_cv(data, labels, subjects):

    def concatenate_data_labels(idxs): # one giant matrix concatenating trials across subjects/sessions. this fxn is called in the one below.
        n_features = data[0].shape[1] # n_trials, n_voxels = data[0].shape 
        n_trials_total = sum([d.shape[0] for k,d in enumerate(data) if k in idxs]) # cant do data[0].shape[0] because some weird sessions have diff # trials
        train_data = np.zeros((n_trials_total, n_features)) # np.zeros((n_trials*(len(loso_indices)), n_voxels))
        train_labels = np.array([])
        start_index = 0 # initialize
        for idx in idxs:
            n_trials = data[idx].shape[0]
            end_index = start_index + n_trials
            train_data[start_index:end_index,:] = data[idx]
            train_labels = np.append(train_labels, labels[idx])
            start_index += n_trials
        return train_data, train_labels 

    def predict_left_out_subject(sub): # train on all but one sub, test on that sub. this fxn is called in the parallel loop.
        sub_indices = [j for j,s in enumerate(subjects) if s == sub]
        loso_indices = [j for j,s in enumerate(subjects) if s != sub]
        
        train_data, train_labels = concatenate_data_labels(loso_indices)
        test_data, test_labels = concatenate_data_labels(sub_indices)

        classifier = LinearSVC(C = 1.0, loss='hinge', dual = 'auto') # this differs slightly from SVC(kernel = 'linear') but converges faster
        classifier = classifier.fit(train_data, train_labels)
        predicted_labels = classifier.predict(test_data)
        acc = sum(predicted_labels == test_labels)/len(predicted_labels)
        return acc 

    accuracies = Parallel(n_jobs = len(np.unique(subjects)))( # 
        delayed(predict_left_out_subject)(sub) for sub in np.unique(subjects)
    )

    return accuracies


def plot_accuracies(tasks, acc_srm, acc_nosrm, save=False):
    task_chance = {
        'goNogo': 1/2,
        'shapeMatching': 1/7,
        'spatialTS': 1/4,
        'cuedTS': 1/4,
        'flanker': 1/2,
        'stopSignal': 1/3,
        'nBack': 1/2,
        'directedForgetting': 1/4
    }
    bar_width = 0.25
    x = np.arange(len(tasks))
    fig, ax = plt.subplots(1,1, figsize = (10,5))
    fig.suptitle('Trial-by-trial task decoding, SRM-transformed vs. MNI-only')
    for i,task in enumerate(tasks):
        x_pair = [x[i] - bar_width/2, x[i]+bar_width/2]
        ax.bar(x_pair, [np.mean(acc_srm[i]), np.mean(acc_nosrm[i])], 
            yerr = [np.std(acc_srm[i]), np.std(acc_nosrm[i])],
            width=bar_width, label = ['SRM-transformed', 'MNI only'], color = ['green', 'blue'], alpha = 0.5, capsize=2)
        x_pair = [x[i] - bar_width*1.5, x[i]+bar_width*1.5]
        ax.hlines(task_chance[task], xmin = x_pair[0], xmax = x_pair[1], color='red', linestyle='--')
    ax.set_xlabel('Tasks')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel('Leave-one-subject-out cross-validation accuracy')
    ax.set_ylim(0,1)
    ax.legend(['SRM-transformed', 'MNI only'],  loc='lower right')
    plt.show()
    if save:
        plt.savefig('/scratch/users/csiyer/decoding_outputs/accuracy_plot')


def run_decoding():
    tasks = ['directedForgetting','stopSignal','nBack','goNogo','shapeMatching','spatialTS','cuedTS','flanker']
    results_srm = []
    results_nosrm = []

    for task in tasks:
        print(f'starting {task}')
        data, events, subjects = load_data(task)
        print(f'loaded data for {task}')

        # data, labels = average_trials(data, events)
        data, labels = label_trs(data, events, task, correct_only=False)
        data_srm = srm_transform(data, subjects)
        print(f'labeled and srm\'d  {task}')

        task_accuracies_srm = loso_cv(data_srm, labels, subjects)
        task_accuracies_nosrm = loso_cv(data, labels, subjects)

        print(f'For {task}, the average LOSO-CV accuracy with SRM is {np.mean(task_accuracies_srm)}')
        print(f'For {task}, the average LOSO-CV accuracy with NO SRM is {np.mean(task_accuracies_nosrm)}')

        results_srm.append(task_accuracies_srm)
        results_nosrm.append(task_accuracies_nosrm)

        del data, data_srm, events, subjects, labels

    np.save('/scratch/users/csiyer/decoding_outputs/results_srm_final.npy', results_srm)
    np.save('/scratch/users/csiyer/decoding_outputs/results_nosrm_final.npy', results_nosrm)
    plot_accuracies(tasks, results_srm, results_nosrm, save=True)


if __name__ == "__main__":
    run_decoding()