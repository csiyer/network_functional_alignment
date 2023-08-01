"""
This script will read in connectomes created in connectivity.py and tested in reliability.ipynb
and derive (connectivity) Shared Response Models (Chen et al 2015, Guntupalli et al. 2018, Nastase et al. 2019).

Shared Response Modelling derives a single shared space, as well as transformation matrices from 
each subject's native anatomical space into the shared space. These transformation matrices contain
information about subject-specific idiosyncratic functional topographics in the cortex, and these are
what we will later apply to task data to see if they generalize to decoidng cognitive control states. 

We parcellate the cortex and derive one SRM per parcel--this imposes an antomical constraint
on the spatial degree to which voxels can affect shared responses. To get all the subjects'
transformation matrices, we concatenate the SRMs for each parcel to create one whole-brain
transformation matrix per subject, which is saved and used in task_decoding.py

NOTE: the SRM derivation code is copied from the BrianIAK library (https://brainiak.org/)

Author: Chris Iyer
Updated: 7/20/23
"""

##### TO-DO: implement parcellation + derive and concatenate separate SRMs on each parcel


import glob
import numpy as np
from scipy import stats
from utils.brainiak import srm

def load_connectomes():
    data_dict = {}
    sub_nums = np.unique([f[f.find('sub'):f.find('sub')+7] for f in glob.glob('output/connectomes/*avg*')])
    for sub in sub_nums:
        data_dict[sub] =  np.load(glob.glob(f'output/connectomes/*{sub}_avg*')[0]) # these connectomes are of the shape (n_voxels x n_connectivity_targets)
    
    data_list = list(data_dict.values()) # also create unlabeled list
    return data_dict, data_list

def load_fake_connectomes():
    sub_nums = ['sub-s01', 'sub-s02', 'sub-s03', 'sub-s04', 'sub-s05']
    data_dict = {}
    for sub in sub_nums:
        data_dict[sub] = np.random.rand(100, 10) # 100 voxels x 10 parcel targets

    data_list = list(data_dict.values()) # also create unlabeled lists
    return data_dict, data_list


def fit_srm(data, n_features=50, n_iter=20):
    shared_model = srm.SRM(n_iter, n_features)
    print('Fitting SRM, may take a minute ...')
    shared_model.fit(data)
    print('SRM has been fit')
    return shared_model


def save_srm_outputs(shared_model, data_dict):
    outpath = 'outputs/srm_transforms/'
    shared_model.save(outpath + 'srm_saved') # can reload with srm.load
    shared_model.s_.tofile(outpath + 'srm_shared_space') # save shared space (shape will be n_features x n_connectivity_targets)

    sub_nums = list(data_dict.keys()) # in same order as data_list and therefore as srm outputs
    for sub_i in range(len(shared_model.w_)):
        shared_model.w_[sub_i].tofile(f'{outpath + sub_nums[sub_i]}_srm_transform') # save transformation matrices (shape will be n_voxels x n_features)


if __name__ == "__main__":
    data_dict, data_list = load_fake_connectomes()
    shared_model = fit_srm(data_list, n_features=2)
    # save_srm_outputs(shared_model, data_dict)
