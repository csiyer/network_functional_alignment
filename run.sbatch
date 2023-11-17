#!/bin/bash
#SBATCH -J network_connectivity_csi
#SBATCH --time=4-00:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=6G
#SBATCH -p bigmem
# Outputs ---------------------------------
#SBATCH -o  log.out
#SBATCH -e log.err
#SBATCH --mail-user=csiyer@stanford.edu
#SBATCH --mail-type=END
# ------------------------------------------

. ${HOME}/.bashrc  
conda activate fa

# python3 connectivity.py
# python3 connectome_average.py # also handled by reliability.ipynb
# python3 srm.py
python3 task_decoding.py