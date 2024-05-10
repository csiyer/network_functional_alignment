"""
This is a copy of task_decoding.py that implements within-subject (leave-one-session-out) classification
instead of across-subject (leave-one-subject-out). This serves as a sanity check-- if classification
is still at chance in this scenario (especially on trial-balanced tasks like flanker), then something in
the data is garbage.

Author: Chris Iyer
Updated: 5/7/24
"""

import glob, pickle, argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from joblib import Parallel, delayed
# /oak/stanford/groups/russpold/data/network_grant/discovery_BIDS_21.0.1/derivatives/glm_data_MNI

def load_files(task, correct_only = False):
    """Load beta maps and trial-wise data extracted in task_beta_mapping.py"""

    GLM_PATH = '/scratch/users/csiyer/glm_outputs/'
    subjects = np.load(GLM_PATH + task + '_subjects.npy')
    with open(f'/scratch/users/csiyer/glm_outputs/{task}_beta_maps.pkl', 'rb') as f:
            data = pickle.load(f)
    f.close()
    with open(f'/scratch/users/csiyer/glm_outputs/{task}_labels.pkl', 'rb') as f:
            labels_all = pickle.load(f)
    f.close()

    if correct_only:
        labels = [ [trial_type for trial_type, correct in session_labels if correct == 'True'] for session_labels in labels_all ]
        data = [ np.array([beta_map for beta_map, (_, correct) in zip(session_data, session_labels) if correct == "True"]) for session_data, session_labels in zip(data, labels_all)]
    else:
        labels = [ [trial_type for trial_type, _ in session_labels] for session_labels in labels_all ]

    return subjects, data, labels


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

    def concatenate_data_labels(idxs): # helper for below. just concatenates data into one giant matrix and labels into one big list. 
        concat_data = np.vstack([data[i] for i in idxs])
        concat_labels = np.array([label for i,session_labels in enumerate(labels) if i in idxs for label in session_labels])
        return concat_data, concat_labels 
    
    def predict_left_out_session(all_idx, loso_idx):
        train_data, train_labels = concatenate_data_labels([idx for idx in all_idx if idx != loso_idx])
        classifier = LinearSVC(C = 1.0, penalty='l1', loss='squared_hinge', class_weight = 'balanced', dual = 'auto').fit(train_data, train_labels)
        predicted_probs = classifier._predict_proba_lr(data[loso_idx])
        if predicted_probs.shape[1] == 2: # binary case, roc_auc_score wants different input
            predicted_probs = np.max(predicted_probs, axis=1)
        auc = roc_auc_score(y_true = labels[loso_idx], y_score = predicted_probs, multi_class='ovr', average='macro')
        return auc
    
    sub_aucs = []
    for sub in np.unique(subjects):
        sub_session_idxs = [i for i,s in enumerate(subjects) if s==sub] # indices for each session of this subject
        # LEAVE ONE SESSION OUT and decode -- parallelized
        sub_aucs = Parallel(n_jobs = 32)( 
            delayed(predict_left_out_session)(sub_session_idxs, session_idx) for session_idx in sub_session_idxs
        ) 
        sub_aucs.append(np.mean(sub_aucs))

    return sub_aucs


def plot_performance(tasks, results, savename, save=False):
    acc_srm = results['aucs_srm']
    acc_nosrm = results['aucs_nosrm']

    bar_width = 0.25
    x = np.arange(len(tasks))
    fig, ax = plt.subplots(1,1, figsize = (10,5))
    fig.suptitle('Trial-by-trial WITHIN-SUBJECT task decoding, SRM-transformed vs. MNI-only')
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
    ax.set_ylabel('Leave-one-session-out classifier ROC-AUC')
    ax.set_ylim(0,1)
    custom_legend = [Patch(color='green', alpha=0.5),  # Green rectangle for 'SRM-transformed'
                    Patch(color='blue', alpha=0.5),   # Blue rectangle for 'MNI only'
                    Line2D([0], [0], color='red', linestyle='--', alpha=0.4)]  # Dashed red line for 'chance'
    ax.legend(custom_legend, ['SRM-transformed', 'MNI only', 'chance ~= 0.5'], loc='lower right')
    plt.show()
    if save:
        plt.savefig(savename+'_plot')


def run_decoding(correct_only):
    tasks = ['flanker','spatialTS','cuedTS','shapeMatching','directedForgetting','goNogo'] # 'stopSignal',
    results = {
        'aucs_srm' : [],
        'aucs_nosrm' : []
    }
    for task in tasks:
        print(f'starting {task}')
        subjects, data, labels = load_files(task, correct_only=correct_only)
        data_srm = srm_transform(data, subjects)
        print(f'loaded data for {task}')

        task_aucs_srm = loso_cv(data_srm, labels, subjects)
        task_aucs_nosrm = loso_cv(data, labels, subjects)
        
        results['aucs_srm'].append(task_aucs_srm)
        results['aucs_nosrm'].append(task_aucs_nosrm)

        del data, data_srm, subjects, labels
    
    savedir = f'/scratch/users/csiyer/decoding_outputs/current/'
    savelabel = 'withinsubject_correctonly' if correct_only else 'withinsubject_alltrials'
    savename = savedir + savelabel
    print('saving data: ', savename)
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