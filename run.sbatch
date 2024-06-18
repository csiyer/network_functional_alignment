#!/bin/bash
#SBATCH -J network_connectivity_csi
#SBATCH --time=23:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=799M
#SBATCH -p russpold
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
# python3 task_beta_mapping.py

# # silly stuff from ChatGPT to process an optional --correct_only flag  #### i.e., type: sbatch run.sbatch --correct_only
# while [[ $# -gt 0 ]]; do
#     key="$1"
#     case $key in
#         --correct_only)
#         CORRECT_ONLY=true
#         shift
#         ;;
#         *)
#         echo "Invalid flag: $1"
#         exit 1
#         ;;
#     esac
# done
# python3 task_decoding.py --correct_only=${CORRECT_ONLY}

# python3 contrast_map_checks/contrast_map_avg.py
python3 contrast_map_checks/conjunction_analysis.py