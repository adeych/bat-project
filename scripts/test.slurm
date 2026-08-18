#!/bin/bash
#SBATCH --time 00:02:00
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

echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

echo "=== DEBUG: CUDA / PyTorch check ==="
nvidia-smi
python3 -c "
import torch
print('torch version:', torch.__version__)
print('torch.cuda.is_available():', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device count:', torch.cuda.device_count())
    print('device name:', torch.cuda.get_device_name(0))
else:
    print('CUDA not available - reason unclear, check build below')
print('torch built with CUDA:', torch.version.cuda)
"
echo "=== END CUDA / PyTorch check ==="

cd /idiap/home/adeych/bat-project/tests

unset LD_LIBRARY_PATH
python3 cluster_test.py --data_dir /idiap/temp/adeych/data

