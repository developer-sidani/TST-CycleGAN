#!/bin/bash
#SBATCH --time=110:00:00
#SBATCH --ntasks=1
#SBATCH --partition=cuda
#SBATCH --gres=gpu:1
#SBATCH --job-name=multi30k_cycle
#SBATCH --mem=40GB
#SBATCH --output=/home/asidani/logs/cyclegan/multi30k_cycle_%j.log
#SBATCH --error=/home/asidani/logs/cyclegan/multi30k_cycle_%j.err


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

echo "[SCRIPT]: Starting Multi30k cycle consistency phase training for ${SRC_LANG}-${TGT_LANG} pair"
send_discord "[${SLURM_JOB_ID}]: Starting Multi30k Cycle Consistency Training (Phase 3) for ${SRC_LANG}-${TGT_LANG}"

PYTHON_PATH="/home/asidani/.conda/envs/cliptrans/bin/python3"

# Define Multi30k specific paths
IMAGES_DIR="/home/asidani/TST-CycleGAN/data/multi30k/images/train"
IMAGES_TEST_DIR="/home/asidani/TST-CycleGAN/data/multi30k/images/test_${TEST_YEAR}_${TEST_SET}"

# Use the processed TSV files created in the caption phase
TEMP_DIR="/home/asidani/TST-CycleGAN/data/multi30k/data/task1/processed"
CAPTION_FILE_SRC="${TEMP_DIR}/train_${SRC_LANG}.tsv"
CAPTION_FILE_TGT="${TEMP_DIR}/train_${TGT_LANG}.tsv"
CAPTION_FILE_SRC_EVAL="${TEMP_DIR}/test_${SRC_LANG}.tsv"
CAPTION_FILE_TGT_EVAL="${TEMP_DIR}/test_${TGT_LANG}.tsv"

# Model names and directories
TRANSLATE_MODEL_NAME="multi30k_${SRC_LANG}_${TGT_LANG}_translate_p2"
CYCLE_MODEL_NAME="multi30k_${SRC_LANG}_${TGT_LANG}_cycle_p3"
PRETRAINED_MODEL="./models/${TRANSLATE_MODEL_NAME}/final/"
SAVE_DIR="./models/${CYCLE_MODEL_NAME}"
mkdir -p $SAVE_DIR

# Define training parameters
CLIP_MODEL="openai/clip-vit-base-patch32"
MBART_MODEL="facebook/mbart-large-50"
PREFIX_LENGTH=10
BATCH_SIZE=8  # Reduced batch size for cycle training
EPOCHS=10
MAPPING_NETWORK="mlp"

echo "[SCRIPT]: Using pretrained translate model from ${PRETRAINED_MODEL}"
echo "[SCRIPT]: Will save cycle model to ${SAVE_DIR}"

TRAIN_PARAMS=(
    --style_a ${SRC_LANG}
    --style_b ${TGT_LANG}
    --training_phase cycle
    --use_clip
    --clip_model_name ${CLIP_MODEL}
    --prefix_length ${PREFIX_LENGTH}
    --mapping_network ${MAPPING_NETWORK}
    --image_dir ${IMAGES_DIR}
    --caption_file_a ${CAPTION_FILE_SRC}
    --caption_file_b ${CAPTION_FILE_TGT}
    --caption_file_a_eval ${CAPTION_FILE_SRC_EVAL}
    --caption_file_b_eval ${CAPTION_FILE_TGT_EVAL}
    --lang ${SRC_LANG}
    --generator_model_tag ${MBART_MODEL}
    --discriminator_model_tag "distilbert-base-multilingual-cased"
    --batch_size ${BATCH_SIZE}
    --epochs ${EPOCHS}
    --from_pretrained ${PRETRAINED_MODEL}
    --save_base_folder ${SAVE_DIR}
    --save_steps 1
    --learning_rate 1e-5
    --lr_scheduler_type "linear"
    --warmup
    --lambdas "1|1|0.5|0.5|10"  # Cycle consistency weight higher
    --use_cuda_if_available
    --comet_logging
)

# Run the training script
$PYTHON_PATH train.py "${TRAIN_PARAMS[@]}"

send_discord "[${SLURM_JOB_ID}]: Multi30k cycle consistency phase training completed for ${SRC_LANG}-${TGT_LANG}"

# Send logs
LOG_FILE="/home/asidani/logs/cyclegan/multi30k_cycle_${SLURM_JOB_ID}.log"
ERR_FILE="/home/asidani/logs/cyclegan/multi30k_cycle_${SLURM_JOB_ID}.err"

python3 /home/asidani/notif.py "$LOG_FILE" "$ERR_FILE" 