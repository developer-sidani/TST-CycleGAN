#!/bin/bash
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --partition=cuda
#SBATCH --gres=gpu:1
#SBATCH --job-name=tst
#SBATCH --mem=40GB
#SBATCH --output=/home/asidani/logs/cyclegan/job_name_%j.log
#SBATCH --error=/home/asidani/logs/cyclegan/job_name_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahmad.sidani@studenti.polito.it

###### 1 Load the module
module load nvidia/cudasdk/11.6
module load intel/python/3

function send_discord {
    python3 /home/asidani/message.py "$@"
}


echo "[SCRIPT]: Checking GPU availability"
which nvidia-smi || echo "nvidia-smi not found"
nvidia-smi || echo "Unable to run nvidia-smi"  

# Select GPU with least memory usage
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{ print NR-1 " " $1 }' | sort -k2 -n | tail -n1 | awk '{ print $1 }')
echo "[SCRIPT]: Selected GPU ID: $CUDA_VISIBLE_DEVICES"