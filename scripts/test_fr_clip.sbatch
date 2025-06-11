#!/bin/bash
#SBATCH --time=00:59:59
#SBATCH --ntasks=1
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --job-name=fr_test_cyclegan
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

source ~/.bashrc

source ~/miniconda3/bin/activate
conda activate cliptrans

cd ~/TST-CycleGAN

send_discord "[${SLURM_JOB_ID}]: Starting Testing"



# Get all checkpoint directories - use the new CLIPTrans checkpoint directory
CKPT_BASE_DIR="/home/asidani/TST-CycleGAN/ckpts_fr_1e5_clip/"

# Check if the checkpoint directory exists
if [ ! -d "$CKPT_BASE_DIR" ]; then
    send_discord "[${SLURM_JOB_ID}]: Error - Checkpoint directory not found: $CKPT_BASE_DIR"
    exit 1
fi

CKPT_DIRS=($(find ${CKPT_BASE_DIR} -maxdepth 1 -type d -name "epoch_*" -printf "%f\n"))

# Check if any checkpoint directories were found
if [ ${#CKPT_DIRS[@]} -eq 0 ]; then
    send_discord "[${SLURM_JOB_ID}]: Error - No epoch directories found in $CKPT_BASE_DIR"
    exit 1
fi

send_discord "[${SLURM_JOB_ID}]: Found ${#CKPT_DIRS[@]} checkpoint directories"

test_ds=("2017_flickr" "2018_flickr" "2017_mscoco")
# test_ds=("2016_flickr")

for ckpt_dir in "${CKPT_DIRS[@]}"; do
    send_discord "[${SLURM_JOB_ID}]: Testing with checkpoint ${ckpt_dir}"
    
    for ds in "${test_ds[@]}"; do
        TEST_PARAMS=(
            --style_a en
            --style_b fr
            --lang en
            --from_pretrained ${CKPT_BASE_DIR}${ckpt_dir}/
            --path_paral_A_test=/home/asidani/TST-CycleGAN/data/multi30k/data/task1/raw/test_$ds.en
            --path_paral_B_test=/home/asidani/TST-CycleGAN/data/multi30k/data/task1/raw/test_$ds.fr
            --path_paral_test_ref=/home/asidani/TST-CycleGAN/data/multi30k/data/task1/raw/fr/$ds/
            --n_references=1
            --generator_model_tag facebook/mbart-large-50
            --discriminator_model_tag distilbert-base-multilingual-cased
            --test_ds $ds
            --max_sequence_length=64 
            --batch_size=16 
            --pin_memory 
            --use_cuda_if_available 
            --comet_logging
            --num_workers 1
            --use_cliptrans
        )
        send_discord "[${SLURM_JOB_ID}]: Testing $ds with checkpoint ${ckpt_dir}"
        python test.py "${TEST_PARAMS[@]}"
    done
done


send_discord "[${SLURM_JOB_ID}]: Testing completed"

LOG_FILE="/home/asidani/logs/cyclegan/job_name_${SLURM_JOB_ID}.log"
ERR_FILE="/home/asidani/logs/cyclegan/job_name_${SLURM_JOB_ID}.err"

python3 /home/asidani/notif.py "$LOG_FILE" "$ERR_FILE"