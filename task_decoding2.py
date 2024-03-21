"""
Alternate version to run RBF kernel while other code runs
"""

import os, glob, pickle, argparse
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.maskers import MultiNiftiMasker
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
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

    data_files = [f for f in glob.glob(bids_dir + f'/**/*{task}*optcom_bold.nii.gz', recursive=True) if 'ses-11' not in f and 'ses-12' not in f] # previously optcomDenoised
    confound_files = [f for f in glob.glob(bids_dir + f'/**/*{task}*confounds*', recursive=True) if 'ses-11' not in f and 'ses-12' not in f]

    def process_confound_files(data_files, confound_files):
        confound_dfs = []
        for d in data_files: # need to index on the data because there are some wonky ones with a confound file but no data file
            sub_ses = d[d.find('sub'):d.find('sub')+14]
            c = [f for f in confound_files if sub_ses in f][0]
            c = pd.read_csv(c, sep='\t')
            c = c[[col for col in c.columns if 'cosine' in col or 'trans' in col or 'rot' in col]] # just get cosine and 24 motion regressors
            confound_dfs.append(c)
        return confound_dfs
    confounds = process_confound_files(data_files, confound_files)

    data = MultiNiftiMasker(
        mask_img = get_combined_mask(), # mask where it's gray matter above 50% and the parcellation applies
        standardize = 'zscore_sample',
        n_jobs = 32
    ).fit_transform(data_files, confounds = confounds)

    events = [pd.read_csv(f, sep='\t') for f in glob.glob(bids_dir + f'/**/*{task}*events*', recursive=True) if 'ses-11' not in f and 'ses-12' not in f] 
    subjects = [e[e.find('sub') : e.find('sub')+7] for e in data_files]

    return data, events, subjects


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

        # classifier = LinearSVC(C = 1.0, penalty='l1', loss='squared_hinge', class_weight = 'balanced', dual = 'auto').fit(train_data, train_labels)
        classifier = SVC(C = 1.0, kernel = 'rbf', class_weight = 'balanced').fit(train_data, train_labels)
        predicted_labels = classifier.predict(test_data)
        predicted_probs = classifier.predict_proba(test_data)
        if predicted_probs.shape[1] == 2: # binary case, roc_auc_score wants different input
            predicted_probs = np.max(predicted_probs, axis=1)

        auc = roc_auc_score(y_true = test_labels, y_score = predicted_probs, multi_class='ovr', average='micro')
        cm = confusion_matrix(test_labels, predicted_labels)
        return auc, cm

    auc_cm = Parallel(n_jobs = min(len(np.unique(subjects)), 32) )( 
        delayed(predict_left_out_subject)(sub) for sub in np.unique(subjects)
    )

    aucs = [s[0] for s in auc_cm]
    cms = [s[1] for s in auc_cm]
    return aucs, cms


def plot_performance(tasks, results, savename, save=False):
    acc_srm = results['aucs_srm']
    acc_nosrm = results['aucs_nosrm']

    bar_width = 0.25
    x = np.arange(len(tasks))
    fig, ax = plt.subplots(1,1, figsize = (10,5))
    fig.suptitle('Trial-by-trial task decoding, SRM-transformed vs. MNI-only')
    for i in range(len(tasks)):
        x_pair = [x[i] - bar_width/2, x[i]+bar_width/2]
        ax.bar(x_pair, [np.mean(acc_srm[i]), np.mean(acc_nosrm[i])], 
            yerr = [np.std(acc_srm[i]), np.std(acc_nosrm[i])],
            width=bar_width, label = ['SRM-transformed', 'MNI only'], color = ['green', 'blue'], alpha = 0.5, capsize=2)
        # x_pair = [x[i] - bar_width*1.5, x[i]+bar_width*1.5]
        # ax.hlines(task_chance[task], xmin = x_pair[0], xmax = x_pair[1], color='red', linestyle='--')
    
    ax.axhline(0.5, color='red', linestyle='--', alpha = 0.4, label = 'chance')
    ax.set_xlabel('Tasks')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation = 30)
    ax.set_ylabel('Leave-one-subject-out classifier ROC-AUC')
    ax.set_ylim(0,1)
    custom_legend = [Patch(color='green', alpha=0.5),  # Green rectangle for 'SRM-transformed'
                    Patch(color='blue', alpha=0.5),   # Blue rectangle for 'MNI only'
                    Line2D([0], [0], color='red', linestyle='--', alpha=0.4)]  # Dashed red line for 'chance'
    ax.legend(custom_legend, ['SRM-transformed', 'MNI only', 'chance ~= 0.5'], loc='lower right')
    plt.show()
    if save:
        plt.savefig(savename+'_plot')


def run_decoding(correct_only):
    tasks = ['stopSignal','nBack','directedForgetting','goNogo','shapeMatching','spatialTS','cuedTS','flanker']
    results = {
        'aucs_srm' : [],
        'aucs_nosrm' : [],
        'cms_srm' : [],
        'cms_nosrm' : []
    }

    for task in tasks:
        print(f'starting {task}')
        data, events, subjects = load_data(task)
        print(f'loaded {len(data)} data files for {task}')

        data, labels = label_trs(data, events, task, correct_only=correct_only) 
        data_srm = srm_transform(data, subjects)
        print(f'labeled and srm\'d  {task}')

        task_aucs_srm, task_cms_srm = loso_cv(data_srm, labels, subjects)
        task_aucs_nosrm, task_cms_nosrm = loso_cv(data, labels, subjects)

        results['aucs_srm'].append(task_aucs_srm)
        results['cms_srm'].append(task_cms_srm)
        results['aucs_nosrm'].append(task_aucs_nosrm)
        results['cms_nosrm'].append(task_cms_nosrm)

        del data, data_srm, events, subjects, labels
    
    savelabel = 'rbf_correctonly' if correct_only else 'rbf_alltrials'
    savedir = f'/scratch/users/csiyer/decoding_outputs/fifth_confounds_{savelabel}/'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    savename = savedir + savelabel
    with open(savename + '.pkl', 'wb') as file:
        pickle.dump(results, file)
    file.close()
    plot_performance(tasks, results, savename, save=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process --correct_only flag')
    parser.add_argument('--correct_only', type=bool, default=False, help='Boolean flag')
    args = parser.parse_args()
    print('correct_only: ', args.correct_only)

    run_decoding(correct_only = args.correct_only)