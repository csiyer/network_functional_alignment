"""
Knits confusion matrices into a readable PDF, and prints results of statistical tests of decoding results
Updated 5/2/24 
"""

import numpy as np
import pickle
import pandas as pd
import statsmodels.formula.api as smf

################# 1. visualize confusion matrices  #################
def aggregate_cms(task_cms):
    return np.sum(task_cms, axis=0) / np.sum(task_cms) 

with open('outputs/decoding/alltrials.pkl', 'rb') as file:
    results = pickle.load(file)
file.close()

with open('utils/task_decoding_conditions.json', 'r') as file:
    task_conditions = eval(file.read())
file.close()

task = 'stopSignal'
i=0
cm_srm = aggregate_cms(results['cms_srm'][i])
cm_nosrm = aggregate_cms(results['cms_nosrm'][i])

labels = sorted( np.unique([i for i in task_conditions[task]['values'].values()]) ) # sorted() should give same label order as the confusion_matrix fxn
if task == 'stopSignal' and cm_srm.shape[0] < 3:
    labels = ['go', 'stop_success']
print(labels)


################# 2. statistical tests on decoding results ##############
cm_srm = [aggregate_cms(task_cms) for task_cms in results['cms_srm']]
cm_nosrm = [aggregate_cms(task_cms) for task_cms in results['cms_nosrm']]
task_order = ['stopSignal','directedForgetting','goNogo','shapeMatching','spatialTS','cuedTS','flanker']

# linear MLM
data = {'method': [], 'task': [], 'auc': []}
for i, (aucs_srm, aucs_nosrm) in enumerate(zip(results['aucs_srm'], results['aucs_nosrm'])):
    for a in aucs_srm:
        data['method'].append('srm')
        data['task'].append(i)
        data['auc'].append(a)
    for a in aucs_nosrm:
        data['method'].append('no_srm')
        data['task'].append(i)
        data['auc'].append(a)

df = pd.DataFrame(data)
df['task'] = df['task'].astype('category')
md = smf.mixedlm("auc ~ method", df, groups=df["task"], re_formula="~method")
mdf = md.fit()
print(mdf.summary())

# t-tests
from scipy.stats import ttest_rel
print('Follow-up t-tests:')
print('All data points, paired (ignore): ', ttest_rel(np.array(results['aucs_srm']).flatten(), np.array(results['aucs_nosrm']).flatten())) # observations not independent b/c from same CV fold
print('Averaged within task (across folds): ', ttest_rel(np.mean(results['aucs_srm'], axis=1), np.mean(results['aucs_nosrm'], axis=1))) # average within task