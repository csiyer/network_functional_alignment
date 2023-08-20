#!/bin/bash
#SBATCH -J network_connectivity_csi
#SBATCH --time=1-00:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH -p russpold
# Outputs ---------------------------------
#SBATCH -o  log.out
#SBATCH -e log.err
#SBATCH --mail-user=csiyer@stanford.edu
#SBATCH --mail-type=END
# ------------------------------------------

module load python/3.11
python sherlock_connectivity.py