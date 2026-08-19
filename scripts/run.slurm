#!/bin/bash
#SBATCH --time 00:10:00
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --partition gpu
#SBATCH --gpus rtx3090:1
#SBATCH --mem 70G
#SBATCH --account oh-ik

#SBATCH --job-name=test_access
#SBATCH --output=/idiap/home/adeych/bat-project/results/slurm_logs/myjob-%A_%a.out
#SBATCH --error=/idiap/home/adeych/bat-project/results/slurm_logs/myjob-%A_%a.err

source /idiap/temp/adeych/miniforge3/bin/activate temp-env-4

cd /idiap/home/adeych/bat-project/tests

python3 encoder_extraction.py --encoder perch2 --num_audio 20