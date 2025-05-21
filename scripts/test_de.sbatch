#!/bin/bash
#SBATCH --time=110:00:00
#SBATCH --ntasks=1
#SBATCH --partition=cuda
#SBATCH --gres=gpu:1
#SBATCH --job-name=de_test_cyclegan
#SBATCH --mem=40GB
#SBATCH --output=/home/asidani/logs/cyclegan/job_name_%j.log
#SBATCH --error=/home/asidani/logs/cyclegan/job_name_%j.err



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

source ~/.bashrc

source activate /home/asidani/.conda/envs/cliptrans

send_discord "[${SLURM_JOB_ID}]: Starting Testing"

PYTHON_PATH="/home/asidani/.conda/envs/cliptrans/bin/python3"



# Get all checkpoint directories
CKPT_BASE_DIR="/home/asidani/TST-CycleGAN/ckpts_de_1e5/"
CKPT_DIRS=($(find ${CKPT_BASE_DIR} -maxdepth 1 -type d -name "epoch_*" -printf "%f\n"))

test_ds=("2016_flickr" "2017_flickr" "2018_flickr" "2017_mscoco")
# test_ds=("2016_flickr")

for ckpt_dir in "${CKPT_DIRS[@]}"; do
    send_discord "[${SLURM_JOB_ID}]: Testing with checkpoint ${ckpt_dir}"
    
    for ds in "${test_ds[@]}"; do
        TEST_PARAMS=(
            --style_a en
            --style_b de
            --lang en
            --from_pretrained ${CKPT_BASE_DIR}${ckpt_dir}/
            --path_paral_A_test=/home/asidani/TST-CycleGAN/data/multi30k/data/task1/raw/test_$ds.en
            --path_paral_B_test=/home/asidani/TST-CycleGAN/data/multi30k/data/task1/raw/test_$ds.de
            --path_paral_test_ref=/home/asidani/TST-CycleGAN/data/multi30k/data/task1/raw/de/$ds/
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
        )
        send_discord "[${SLURM_JOB_ID}]: Testing $ds with checkpoint ${ckpt_dir}"
        $PYTHON_PATH test.py "${TEST_PARAMS[@]}"
    done
done


send_discord "[${SLURM_JOB_ID}]: Testing completed"

LOG_FILE="/home/asidani/logs/cyclegan/job_name_${SLURM_JOB_ID}.log"
ERR_FILE="/home/asidani/logs/cyclegan/job_name_${SLURM_JOB_ID}.err"

python3 /home/asidani/notif.py "$LOG_FILE" "$ERR_FILE"