#!/bin/bash
#SBATCH -J network_connectivity_csi3
#SBATCH --time=23:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8G
#SBATCH -p bigmem
# Outputs ---------------------------------
#SBATCH -o  log3.out
#SBATCH -e log3.err
#SBATCH --mail-user=csiyer@stanford.edu
#SBATCH --mail-type=END
# ------------------------------------------

. ${HOME}/.bashrc  
conda activate fa

# python3 connectivity.py
# python3 connectome_average.py # also handled by reliability.ipynb
# python3 srm.py

# silly stuff from ChatGPT to process an optional --correct_only flag  #### i.e., type: sbatch run.sbatch --correct_only
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --correct_only)
        CORRECT_ONLY=true
        shift
        ;;
        *)
        echo "Invalid flag: $1"
        exit 1
        ;;
    esac
done
python3 task_decoding_rbf.py --correct_only=${CORRECT_ONLY}