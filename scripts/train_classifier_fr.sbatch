#!/bin/bash
#SBATCH --time=23:59:59
#SBATCH --ntasks=1
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --job-name=classif-cyclegan
#SBATCH --mem=40GB
#SBATCH --output=/home/asidani/logs/cyclegan/job_name_%j.log
#SBATCH --error=/home/asidani/logs/cyclegan/job_name_%j.err


function send_discord {
    python3 /home/asidani/message.py "$@"
}


echo "[SCRIPT]: Checking GPU availability"
which nvidia-smi || echo "nvidia-smi not found"
nvidia-smi || echo "Unable to run nvidia-smi"  

# Select GPU with least memory usage
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{ print NR-1 " " $1 }' | sort -k2 -n | tail -n1 | awk '{ print $1 }')
echo "[SCRIPT]: Selected GPU ID: $CUDA_VISIBLE_DEVICES"


source ~/miniconda3/bin/activate
conda activate cliptrans

cd ~/TST-CycleGAN

send_discord "[${SLURM_JOB_ID}]: Starting Training"


TRAIN_PARAMS_CLASSIFIER=(
    --dataset_path "./data/multi30k/data/task1/raw"
    --lang1 "en"
    --lang2 "fr" 
    --max_sequence_length 64 
    --batch_size 64
    --use_cuda_if_available 
    --learning_rate 5e-5 
    --epochs 10 
    --lr_scheduler_type "linear" 
    --model_tag "distilbert-base-multilingual-cased" 
    --save_base_folder "/home/asidani/TST-CycleGAN/classifier_fr/"
    --save_steps 1 
    --eval_strategy "epoch" 
    --eval_steps 1 
    --comet_logging
)

python -m utils.train_classifier.py "${TRAIN_PARAMS_CLASSIFIER[@]}"


send_discord "[${SLURM_JOB_ID}]: Train completed"


LOG_FILE="/home/asidani/logs/cyclegan/job_name_${SLURM_JOB_ID}.log"
ERR_FILE="/home/asidani/logs/cyclegan/job_name_${SLURM_JOB_ID}.err"

python3 /home/asidani/notif.py "$LOG_FILE" "$ERR_FILE"