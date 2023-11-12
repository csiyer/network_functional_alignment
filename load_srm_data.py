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

    np.savez(f'/scratch/users/csiyer/decoding_outputs/srm_data_{task}.npz', data=data_srm, labels=labels, subjects=subjects)