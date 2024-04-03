"""Mostly for testing purposes. Just to run code on SRM'd data easily."""

import numpy as np
from task_decoding import load_data, na_check, label_trs, srm_transform

tasks = ['goNogo','shapeMatching','spatialTS','cuedTS','flanker','nBack','stopSignal', 'directedForgetting']
for task in tasks:
    print(f'starting {task}')
    data, events, subjects = load_data(task)
    print(f'loaded data for {task}')

    # nas = na_check(data, subjects)
    # data, labels = average_trials(data, events)
    data, labels = label_trs(data, events)
    print(f'labeled {task}')

    data_srm = srm_transform(data, subjects)
    print(f'srm\'d  {task}')

    np.savez(f'/scratch/users/csiyer/decoding_outputs/srm_data_{task}.npz', *data_srm)
    np.savez(f'/scratch/users/csiyer/decoding_outputs/labels_{task}.npz', *labels)
    np.save(f'/scratch/users/csiyer/decoding_outputs/subjects_{task}.npz', subjects)

"""
Reading data back out:

load = np.load(f'/scratch/users/csiyer/decoding_outputs/srm_data_{task}.npz')
data_srm = [load[k] for k in load]
load = np.load(f'/scratch/users/csiyer/decoding_outputs/labels_{task}.npz')
labels = [load[k] for k in load]
subjects = np.load(f'/scratch/users/csiyer/decoding_outputs/subjects_{task}.npz')
"""