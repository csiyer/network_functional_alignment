# Network R01 Functional Alignment Analyses
## Authors: Chris Iyer
Poldrack Lab, Stanford University
Updated: 8/7/2024

This respository contains code + results for functional alignment, task decoding, and conjunction analysis for our Network R01 rest data (discovery sample). To replicate the analyses, run the following scripts:

1. `connectivity.py`: loads rest data from /oak/stanford/groups/russpold/data/network_grant/discovery_BIDS_21.0.1/derivatives/glm_data_MNI. Derives a voxel-to-parcel connectivity matrix for each session of rest data (each of 5 subjects have 10-12 rest sessions). This means each voxel is correlated to each parcel of the 400-dimension Schaefer parcellation (found in `data/templates`; alternatively can pass `atlast='DiFuMo` to the `get_parcellation()` function). Saves connectomes to `outputs/connectomes`.
2. `connectome_average.py`: averages each subject's 12 session-wise connectomes into 1 per subject. We first used `reliability.ipynb` to visualize the connectomes and verify that we see within-subject, across-session shared structure--this was clear, so we averaged within-subject.
3. `srm.py`: derives connectivity-based Shared Response Models (SRM; [Chen et al. 2015](https://proceedings.neurips.cc/paper_files/paper/2015/file/b3967a0e938dc2a6340e258630febd5a-Paper.pdf); [Nastase et al. 2020](https://www.sciencedirect.com/science/article/pii/S1053811920303517?via%3Dihub#bib92); [Guntupalli et al. 2018](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.106120)). This derives one SRM per parcel (as an anatomical constraint), and concatenates across all parcels. The derivation is done on all subjects' connectomes at once (none are left out--later, in `contrast_map_analyses/conjunction_analysis.py`, this is done with one subject left out at a time). It saves a parcelwise shared response, and subject-specific transformation matrices (concatenated across parcels, from each subject's native voxel space into the shared space). This is our functional alignment method of choice.
4. `task_beta_mapping.py`: models trial-level activation during our battery of cognitive control tasks. These trial maps will then be decoded in the next script. NOTE: the failure of our task decoding to get reliable signal may be because our trials are too fast to get reliable trial-level activation estimates in this script.
5. `task_decoding.py`: decodes trial labels of the trial activation maps produced above, comparing linear-SVC-based decoding performance on only-anatomically-aligned data to SRM-transformed (i.e. functionally aligned) data. Outputs (a .pkl file of classifier accuracy values and a .png plot of classifier ROC-AUC on each task) are saved to `outputs/decoding`.
6. `visualize_results.ipynb`: knits confusion matrices from decoding into a PDF and runs statistical tests on classifier performance data. Not strictly necessary but you can take a look!
7. ALTERNATE ANALYSIS: the purpose of the task decoding analysis is to quantify the additional signal within functionally-aligned task data by decoding task states and comparing to un-aligned data. As a total alternative, I implemented a conjunction analysis of aligned and un-aligned task contrast maps in `contrast_map_analyses/conjunction_analysis.py`. This script re-derives SRM transformations leaving one subject out of the initial derivation. It then transforms all other subjects' task contrast maps into the left-out subject's native space, and computes the Dice coefficient and Pearson R of all pairs of subjects' contrast maps. It also computes these for MNI-only (non-SRMd) data. This offers another way of assessing the benefit of SRM. Results are saved to `outputs/conjunction_analysis`--both the raw coefficients and a plot.

Additional files/folders:
- `environment.yml` contains my conda environment.
- `run.sbatch` contains a sample sbatch script to run all or some of the python scripts on Sherlock, with the memory parameters I used.
- `archive` contains old code and sanity checks I ran, including some contrast map reliability checks and a script performing within-subject decoding.


[Google doc explaining steps in more detail](https://docs.google.com/document/d/13P4QTHxrT5lZfCOXtN59xCKpJfnObtqh3uZkuRqPxR4/edit?pli=1#heading=h.2qncjqtc0b5j)

Please contact c.iyer@columbia.edu for all questions!
