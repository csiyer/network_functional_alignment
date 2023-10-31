"""
Here, we take the subject voxel-to-feature transformation matrices derived in srm.py 
to transform task data into the shared space.

Then, we decode trial conditions (e.g., congruent vs. incongruent, go vs. no-go) between subjects using leave-one-subject-out cross-validation,
with and without the SRM transformation. In other words, a classifier is trained on SRM-transformed data from all sessions of all
other subjects on a given task. Then, for each trial of the left-out subjects 5 repetitions of that task, it produces a prediction for each trial,
which we average to get an accuracy score for that fold. Finally, accuracies are averaged across folds/left-out subjects.

Assessing the performance benefit of SRM transformation tests how functionally shared or idiosyncratic
neural signatures of these cognitive control tasks are.

Notes:
- Currently, the TRs corresponding to each trial are averaged, and this average image is the basis of trial type prediction. 
    We could change this to decode each TR, or to do something more advanced like decode on GLM outputs.
- NEXT STEP:
    Differentiate trial types based on behavioral outcome (correct/incorrect response)

Other ideas: 
- Manipulate what data we derive the SRM from. Does GNG-based alignment help decode stop-signal data? Etc.
- Derive SRM from contrast maps instead of rest connectivity?

Author: Chris Iyer
Updated: 10/30/23
"""

import glob
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.svm import LinearSVC
import nibabel as nib
from nilearn.maskers import MultiNiftiMasker
import matplotlib.pyplot as plt

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

    data_files = glob.glob(bids_dir + f'/**/*{task}*optcomDenoised*nii.gz', recursive=True) # should be ~25 (5 subjects x 5ish sessions each)
    # confound_files = glob.glob(bids_dir + f'/**/*{task}*confounds*', recursive=True) 
    event_files = glob.glob(bids_dir + f'/**/*{task}*events*', recursive=True)

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

    print('data shape check:')
    for d,l in zip(data_trialaveraged, labels):
        print((d.shape, l.shape))
    return data_trialaveraged, labels

def srm_transform(data, subjects):
    data_srm = data.copy()
    srm_dir = '/scratch/users/csiyer/srm_outputs/'
    srm_files = glob.glob(srm_dir + '*transform*')
    for i,sub in enumerate(subjects):
        srm_transform = np.load([s for s in srm_files if sub in s][0])
        data_srm[i] = np.dot(data[i], srm_transform)
    return data_srm

def loso_cv(data, labels, subjects):

    accuracies = np.zeros(len(np.unique(subjects))) # one for each left-out subject
    
    for i,sub in enumerate(np.unique(subjects)):
        loso_indices = [j for j,s in enumerate(subjects) if s != sub] # leave out one subject

        # concatenate one huge training data matrix of samples x features (i.e. [trials x sessions] x voxels)
        n_trials, n_voxels = data[0].shape
        train_data = np.zeros((n_trials*(len(subjects)-1), n_voxels))
        train_labels = np.array([])
        for j,loso_idx in enumerate(loso_indices):
            start_index = j*n_trials
            end_index = start_index + n_trials
            train_data[start_index:end_index,:] = data[loso_idx]
            train_labels = np.append(train_labels, labels[loso_idx])

        print((train_data.shape, train_labels.shape))
            
        # fit support vector classifier
        classifier = LinearSVC(C = 1.0, loss='hinge') # this differs slightly from SVC(kernel = 'linear') but converges faster
        classifier = classifier.fit(train_data, train_labels)

        # Predict on the left out subject
        predicted_labels = classifier.predict(data[i])
        accuracies[i] = sum(predicted_labels == labels[i])/len(predicted_labels)

    return accuracies

def plot_accuracies(acc_srm, acc_nosrm, save=False):
    tasks = ['goNogo','shapeMatching','spatialTS','cuedTS','directedForgetting','flanker','nBack','stopSignal']

    bar_width = 0.25
    x = np.arange(len(tasks))

    fig, ax = plt.subplots(1,1, figsize = (10,5))
    fig.suptitle('Trial-by-trial task decoding, SRM-transformed vs. MNI-only')
    for i in range(len(tasks)):
        x_pair = [x[i] - bar_width/2, x[i]+bar_width/2]
        ax.bar(x_pair, [np.mean(acc_srm), np.mean(acc_nosrm)], 
            yerr = [np.std(acc_srm), np.std(acc_nosrm)],
            width=bar_width, label = ['SRM-transformed', 'MNI only'], color = ['red', 'blue'], alpha = 0.5, capsize=2)

    ax.set_xlabel('Tasks')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel('Leave-one-subject-out cross-validation accuracy')
    ax.set_ylim(0,1)
    ax.legend(['SRM-transformed', 'MNI only'],  loc='lower right')

    plt.show()
    if save:
        plt.savefig('/scratch/users/csiyer/decoding_outputs/fig1')


def run_decoding():
    tasks = ['goNogo','shapeMatching','spatialTS','cuedTS','directedForgetting','flanker','nBack','stopSignal']
    accuracies_srm = []
    accuracies_nosrm = []

    for task in tasks:
        print(f'starting {task}')
        data, events, subjects = load_data(task)
        print(f'loaded data for {task}')
        data, labels = average_trials(data, events)
        print(f'averaged data for {task}')
        data_srm = srm_transform(data, subjects)
        print(f'srm\'d data for {task}')

        task_accuracies_srm = loso_cv(data_srm, labels, subjects)
        task_accuracies_nosrm = loso_cv(data, labels, subjects)

        print(f'For {task}, the average LOSO-CV accuracy with SRM is {np.mean(task_accuracies_srm)}')
        print(f'For {task}, the average LOSO-CV accuracy with NO SRM is {np.mean(task_accuracies_nosrm)}')

        accuracies_srm.append(task_accuracies_srm)
        accuracies_nosrm.append(task_accuracies_nosrm)

    plot_accuracies(accuracies_srm, accuracies_nosrm, save=True)
    np.save(accuracies_srm, '/scratch/users/csiyer/decoding_outputs/acc_srm.npy')
    np.save(accuracies_nosrm, '/scratch/users/csiyer/decoding_outputs/acc_nosrm.npy')

    return accuracies_srm, accuracies_nosrm


if __name__ == "__main__":
    run_decoding()