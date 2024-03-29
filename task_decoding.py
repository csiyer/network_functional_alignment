"""
****NEW**** this data first extracts trial-wise beta series maps using first-level GLM!

Here, we use transformation matrices from srm.py to transform task data into the shared space.

Then, we decode trial conditions (e.g., congruent vs. incongruent; see full condition list below) 
between subjects using leave-one-subject-out cross-validation, with and without the SRM transformation. 
In each fold, all of one subject's sessions are excluded from the training set and tested on, using classifiers trained
on either SRM'd or raw data from all other subjects. Each TR is classified according to its trial condition.

Assessing the performance benefit of SRM transformation tests how functionally shared or idiosyncratic
neural signatures of these cognitive control tasks are.

Author: Chris Iyer
Updated: 3/29/24
"""

import os, glob, pickle, argparse
import numpy as np
import pandas as pd
from nilearn.maskers import MultiNiftiMasker
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import concat_imgs
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
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

def glm_lsa(data_files, events, confounds, subjects, glm_params, correct_only=False):
    """Trial-wise GLM modeling, using the Least Squares - All approach"""
    beta_maps = []
    labels = []

    for i_sub,sub in enumerate(subjects): # for each session (subjects are repeated)
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
        lsa_glm.fit(data_files[i_sub], events=lsa_events_df, confounds=confounds[i_sub])

        # extract beta series maps
        for trial in lsa_events_df['trial_type'].unique():
            beta_map = lsa_glm.compute_contrast(trial, output_type='effect_size') 
            sub_beta_maps.append(beta_map)
            sub_labels.append(trial.split('__')[0]) # original trial_type

        beta_maps.append(concat_imgs(sub_beta_maps))
        labels.append(sub_labels)

    return beta_maps, labels


def glm_lss(data_files, events, confounds, subjects, glm_params, correct_only=False):
    """Trial-wise GLM modeling, using the Least Squares - Separate approach"""
    beta_maps = []
    labels = []

    def label_one_row(df, row_number):
        """Label one trial for one LSS model. Takes events file and row number of trial to model."""
        df = df.copy()

        # Determine which number trial it is *within its condition*
        trials_of_this_type = df["trial_type"] == df.loc[row_number, "trial_type"]
        trial_type_index_list = df["trial_type"].loc[trials_of_this_type].index.tolist()
        trial_number = trial_type_index_list.index(row_number)

        trial_name = f"{trial_condition}__{trial_number:03d}" # new trial-specific label
        df.loc[row_number, "trial_type"] = trial_name
        return df, trial_name

    for i_sub,sub in enumerate(subjects): # for each session (subjects are repeated)
        sub_beta_maps = []
        sub_labels = []
        glm_params['subject_label'] = sub

        for i_trial in range(len(events[i_sub])):
            lss_events, trial_condition = label_one_row(events[i_sub], i_trial)

            lss_glm = FirstLevelModel(**glm_params)
            lss_glm.fit(data_files[i_sub], events=lss_events, confounds=confounds[i_sub])
            beta_map = lss_glm.compute_contrast(trial_condition, output_type="effect_size")

            sub_beta_maps.append(beta_map)
            sub_labels.append(trial_condition.split('__')[0]) # recover original trial name

        beta_maps.append(concat_imgs(sub_beta_maps))
        labels.append(sub_labels)

    return beta_maps, labels


def extract_beta_series(data_files, events, confounds, subjects, correct_only=False, method='LSA'):
    """
    For a given task, this function will run first level GLM to extract trial-wise beta timeseries (LSS or LSA)
        for each subject/session. It will then vectorize the beta maps with nilearn maskers.
    Returns:
        data: a list (of length n_sessions) of beta series data from each trial
        labels: a list (of length n_sessions) of lists of trial_type labels corresponding to data
    """
    glm_params = { # SHOULD PULL THESE SPECIFICALLY FROM SOMEWHERE?
        't_r': 1.49,
        'mask_img': get_combined_mask(),
        'noise_model': 'ar1', #???
        'standardize': False,
        'drift_model': None,
        'smoothing_fwhm': 5, #????
        'minimize_memory': False,
        'n_jobs': 32
    }

    if method=='LSA':
        beta_maps, labels = glm_lsa(data_files, events, confounds, subjects, glm_params, correct_only=correct_only)
    elif method=='LSS':
        beta_maps, labels = glm_lss(data_files, events, confounds, subjects, glm_params, correct_only=correct_only)

    data = MultiNiftiMasker(
        mask_img = get_combined_mask(), # mask where gray matter above 50% and the parcellation applies
        standardize = 'zscore_sample',
        n_jobs = 32
    ).fit_transform(beta_maps)

    return data, labels


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

        classifier = LinearSVC(C = 1.0, penalty='l1', loss='squared_hinge', class_weight = 'balanced', dual = 'auto').fit(train_data, train_labels)
        # classifier = SVC(C = 1.0, kernel = 'rbf', class_weight = 'balanced').fit(train_data, train_labels)
        predicted_labels = classifier.predict(test_data)
        predicted_probs = classifier._predict_proba_lr(test_data)
        if predicted_probs.shape[1] == 2: # binary case, roc_auc_score wants different input
            predicted_probs = np.max(predicted_probs, axis=1)

        auc = roc_auc_score(y_true = test_labels, y_score = predicted_probs, multi_class='ovr', average='macro')
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
        data_files, events, confounds, subjects = load_files(task)
        print(f'loaded {len(data_files)} data files for {task}')

        data, labels = extract_beta_series(data_files, events, confounds, subjects, correct_only=correct_only, method='LSA')
        data_srm = srm_transform(data, subjects)
        print(f'extracted and srm\'d  {task}')

        task_aucs_srm, task_cms_srm = loso_cv(data_srm, labels, subjects)
        task_aucs_nosrm, task_cms_nosrm = loso_cv(data, labels, subjects)

        results['aucs_srm'].append(task_aucs_srm)
        results['cms_srm'].append(task_cms_srm)
        results['aucs_nosrm'].append(task_aucs_nosrm)
        results['cms_nosrm'].append(task_cms_nosrm)

        del data, data_srm, events, confounds, subjects, labels
    
    savelabel = 'correctonly' if correct_only else 'alltrials'
    savedir = f'/scratch/users/csiyer/decoding_outputs/current_{savelabel}/'
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