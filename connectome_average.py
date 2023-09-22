""" 
This script takes the connectivity matrices calculated in connectivity.py and averages within-subject.
Writes 5 session-averaged connectomes (1 per subject) to output. 

Author: Chris Iyer
Updated: 9/22/2023
"""

import glob
import numpy as np



def load_connectomes(path):

    data_dict = {}
    for sub in np.unique([f[f.find('sub'):f.find('sub')+7] for f in glob.glob(path + '*ses*')]):
        data_dict[sub] = {}
        for ses in np.unique([f[f.find('ses'):f.find('ses')+6] for f in glob.glob(path + sub + '*ses*')]):
            curr = np.load(glob.glob(f'{path}*{sub}_{ses}*')[0])
            data_dict[sub][ses] = {
                'connectome': curr, # save memory without this
                # 'connectome_flat': curr.flatten()
            }
    
    return data_dict

if __name__ == "__main__":
    print('beginning averaging')
    data_dict = load_connectomes('/scratch/users/csiyer/')
    for sub in data_dict.keys():
        connectome_avg = np.nanmean([data_dict[sub][ses]['connectome'] for ses in data_dict[sub].keys()], axis=0)
        np.save(f"/scratch/users/csiyer/{sub}_connectome_avg.npy", connectome_avg)
    print('done averaging')