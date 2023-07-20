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
