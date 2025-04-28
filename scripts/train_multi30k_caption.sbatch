#!/bin/bash
#SBATCH --time=110:00:00
#SBATCH --ntasks=1
#SBATCH --partition=cuda
#SBATCH --gres=gpu:1
#SBATCH --job-name=multi30k_caption
#SBATCH --mem=40GB
#SBATCH --output=/home/asidani/logs/cyclegan/multi30k_caption_%j.log
#SBATCH --error=/home/asidani/logs/cyclegan/multi30k_caption_%j.err


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

echo "[SCRIPT]: Starting Multi30k caption phase training for ${SRC_LANG}-${TGT_LANG} pair"
send_discord "[${SLURM_JOB_ID}]: Starting Multi30k Caption Training (Phase 1) for ${SRC_LANG}-${TGT_LANG}"

PYTHON_PATH="/home/asidani/.conda/envs/cliptrans/bin/python3"

# Define Multi30k specific paths
IMAGES_DIR="/home/asidani/TST-CycleGAN/data/multi30k/images/train"
IMAGES_TEST_DIR="/home/asidani/TST-CycleGAN/data/multi30k/images/test_${TEST_YEAR}_${TEST_SET}"

# Convert raw text files to TSV format with image paths
TEMP_DIR="/home/asidani/TST-CycleGAN/data/multi30k/data/task1/processed"
mkdir -p $TEMP_DIR

# Create TSV files from raw files and image_splits
TRAIN_SPLITS="/home/asidani/TST-CycleGAN/data/multi30k/data/task1/image_splits/train.txt"
TEST_SPLITS="/home/asidani/TST-CycleGAN/data/multi30k/data/task1/image_splits/test_${TEST_YEAR}_${TEST_SET}.txt"

CAPTION_FILE_SRC="${TEMP_DIR}/train_${SRC_LANG}.tsv"
CAPTION_FILE_TGT="${TEMP_DIR}/train_${TGT_LANG}.tsv"
CAPTION_FILE_SRC_EVAL="${TEMP_DIR}/test_${SRC_LANG}.tsv"
CAPTION_FILE_TGT_EVAL="${TEMP_DIR}/test_${TGT_LANG}.tsv"

# Generate the TSV files if they don't exist
if [ ! -f "$CAPTION_FILE_SRC" ]; then
    echo "[SCRIPT]: Generating TSV files for training data"
    # Source language
    paste -d '\t' $TRAIN_SPLITS "/home/asidani/TST-CycleGAN/data/multi30k/data/task1/raw/train.${SRC_LANG}" > $CAPTION_FILE_SRC
    # Target language
    paste -d '\t' $TRAIN_SPLITS "/home/asidani/TST-CycleGAN/data/multi30k/data/task1/raw/train.${TGT_LANG}" > $CAPTION_FILE_TGT
fi

if [ ! -f "$CAPTION_FILE_SRC_EVAL" ]; then
    echo "[SCRIPT]: Generating TSV files for test data"
    # Source language
    paste -d '\t' $TEST_SPLITS "/home/asidani/TST-CycleGAN/data/multi30k/data/task1/raw/test_${TEST_YEAR}_${TEST_SET}.${SRC_LANG}" > $CAPTION_FILE_SRC_EVAL
    # Target language
    paste -d '\t' $TEST_SPLITS "/home/asidani/TST-CycleGAN/data/multi30k/data/task1/raw/test_${TEST_YEAR}_${TEST_SET}.${TGT_LANG}" > $CAPTION_FILE_TGT_EVAL
fi

# Model name and save directory
MODEL_NAME="multi30k_${SRC_LANG}_${TGT_LANG}_caption_p1"
SAVE_DIR="./models/${MODEL_NAME}"
mkdir -p $SAVE_DIR

# Define training parameters
CLIP_MODEL="openai/clip-vit-base-patch32"
PREFIX_LENGTH=10
BATCH_SIZE=16
EPOCHS=5
MAPPING_NETWORK="mlp"

TRAIN_PARAMS=(
    --style_a ${SRC_LANG}
    --style_b ${TGT_LANG}
    --training_phase caption
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
    --batch_size ${BATCH_SIZE}
    --epochs ${EPOCHS}
    --save_base_folder ${SAVE_DIR}
    --save_steps 1
    --learning_rate 2e-5
    --lr_scheduler_type "linear"
    --warmup
    --use_cuda_if_available
    --comet_logging
)

# Run the training script
$PYTHON_PATH train.py "${TRAIN_PARAMS[@]}"

send_discord "[${SLURM_JOB_ID}]: Multi30k caption phase training completed for ${SRC_LANG}-${TGT_LANG}"

# Send logs
LOG_FILE="/home/asidani/logs/cyclegan/multi30k_caption_${SLURM_JOB_ID}.log"
ERR_FILE="/home/asidani/logs/cyclegan/multi30k_caption_${SLURM_JOB_ID}.err"

python3 /home/asidani/notif.py "$LOG_FILE" "$ERR_FILE" 