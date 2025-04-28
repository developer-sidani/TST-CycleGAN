#!/bin/bash
#SBATCH --time=110:00:00
#SBATCH --ntasks=1
#SBATCH --partition=cuda
#SBATCH --gres=gpu:1
#SBATCH --job-name=multi30k_train_cyclegan
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

# Activate conda environment
source activate /home/asidani/.conda/envs/cliptrans

# Define language pair and dataset parameters
SRC_LANG=${1:-en}  # Default to English if not provided
TGT_LANG=${2:-de}  # Default to German if not provided
TEST_YEAR=${3:-2016}  # Default to 2016 test set if not provided
TEST_SET=${4:-flickr}  # Default to flickr test set if not provided

echo "[SCRIPT]: Starting Multi30k training for ${SRC_LANG}-${TGT_LANG} pair, test set: ${TEST_YEAR}_${TEST_SET}"
send_discord "[${SLURM_JOB_ID}]: Starting Multi30k Training for ${SRC_LANG}-${TGT_LANG}"

PYTHON_PATH="/home/asidani/.conda/envs/cliptrans/bin/python3"

# Define Multi30k specific paths
IMAGES_DIR="/home/asidani/TST-CycleGAN/data/multi30k/images/train"
IMAGES_TEST_DIR="/home/asidani/TST-CycleGAN/data/multi30k/images/test_${TEST_YEAR}_${TEST_SET}"
TRAIN_SRC="/home/asidani/TST-CycleGAN/data/multi30k/data/task1/raw/train.${SRC_LANG}"
TRAIN_TGT="/home/asidani/TST-CycleGAN/data/multi30k/data/task1/raw/train.${TGT_LANG}"
TEST_SRC="/home/asidani/TST-CycleGAN/data/multi30k/data/task1/raw/test_${TEST_YEAR}_${TEST_SET}.${SRC_LANG}"
TEST_TGT="/home/asidani/TST-CycleGAN/data/multi30k/data/task1/raw/test_${TEST_YEAR}_${TEST_SET}.${TGT_LANG}"
REF_PATH="/home/asidani/TST-CycleGAN/data/multi30k/data/task1/raw/val_${TGT_LANG}/"

TRAIN_PARAMS=(
    --style_a ${SRC_LANG}
    --style_b ${TGT_LANG}
    --lang ${SRC_LANG}
    --path_mono_A ${TRAIN_SRC}
    --path_mono_B ${TRAIN_TGT}
    --path_paral_A_eval ${TEST_SRC}
    --path_paral_B_eval ${TEST_TGT}
    --path_paral_eval_ref ${REF_PATH}
    --n_references 1
    --shuffle
    --generator_model_tag facebook/mbart-large-50
    --discriminator_model_tag distilbert-base-multilingual-cased
    --lambdas "1|1|1|1|0"
    --epochs 30
    --learning_rate 3e-5
    --max_sequence_length 64
    --batch_size 16
    --num_workers 1
    --save_base_folder "./ckpts_multi30k_${SRC_LANG}_${TGT_LANG}/"
    --save_steps 1
    --eval_strategy epochs
    --eval_steps 1
    --pin_memory
    --use_cuda_if_available
    --comet_logging
    # Image-specific parameters for Multi30k
    --image_dir ${IMAGES_DIR}
    --image_dir_eval ${IMAGES_TEST_DIR}
    --use_images
)

# Run the training script
$PYTHON_PATH train.py "${TRAIN_PARAMS[@]}"

send_discord "[${SLURM_JOB_ID}]: Multi30k training completed for ${SRC_LANG}-${TGT_LANG}"

# Send logs
LOG_FILE="/home/asidani/logs/cyclegan/job_name_${SLURM_JOB_ID}.log"
ERR_FILE="/home/asidani/logs/cyclegan/job_name_${SLURM_JOB_ID}.err"

python3 /home/asidani/notif.py "$LOG_FILE" "$ERR_FILE" 