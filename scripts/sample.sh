#!/bin/bash
#SBATCH --time 1:00:00
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --partition gpu
#SBATCH --gpus rtx3090:1
#SBATCH --mem 70G
#SBATCH --account oh-ik

#SBATCH --job-name=test
#SBATCH --output=/idiap/home/adeych/bat-project/results/slurm_logs/myjob-%A_%a.out
#SBATCH --error=/idiap/home/adeych/bat-project/results/slurm_logs/myjob-%A_%a.err

nvidia-smi

eval "$(/idiap/temp/adeych/miniforge3/bin/conda shell.bash hook)"
conda activate temp-env-1

cd /idiap/home/adeych/bat-project/tests

python3 cluster_test.py --data_dir /idiap/temp/adeych/data

