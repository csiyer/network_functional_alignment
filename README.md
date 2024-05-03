# Network R01 Functional Alignment Analyses
## Authors: Chris Iyer
Poldrack Lab, Stanford University

This repository contains code for functional alignment and task decoding analyses for our Network R01 Discovery Sample rest data. The scripts should be run in the following order:

1. `connectivity.py`: derives voxel-to-parcel connectivity matrices for each rest session (each of 5 subjects undergo 12 rest sessions)
2. `reliability.ipynb`: visualizes within-subject, across-session reliability of these connectomes; averages connectomes within subject
        ALT: if just want to average connectomes, run `connectome_average.py`
3. `srm.py`: derives connectivity-based Shared Response Models (SRM; [Chen et al. 2015](https://proceedings.neurips.cc/paper_files/paper/2015/file/b3967a0e938dc2a6340e258630febd5a-Paper.pdf); [Nastase et al. 2020](https://www.sciencedirect.com/science/article/pii/S1053811920303517?via%3Dihub#bib92); [Guntupalli et al. 2018](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006120)). This derives one SRM per parcel (as an anatomical constraint), and saves a parcelwise shared response matrix, as well as subject-specific voxel-to-feature transformation matrices. These matrices transform data from our subjects' native voxel space into the shared space--this is our functional alignment method.
4. `task_beta_mapping.py`: models trial-level activation during cognitive control tasks
5. `task_decoding.py`: decodes task labels of the trial activation estimates producted above, comparing decoding performance with SRM-transformed (i.e., functionally aligned) data to only-anatomically-aligned data.
6. `visualize_results.ipynb`: knits confusion matrices into a PDF and runs statistical tests on classifier performacne data

