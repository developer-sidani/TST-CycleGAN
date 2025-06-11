#!/bin/bash
#SBATCH --time=119:59:59
#SBATCH --ntasks=1
#SBATCH --partition=gpu_a40_ext
#SBATCH --gres=gpu:1
#SBATCH --job-name=clip_cycle_fr_parallel
#SBATCH --mem=40GB
#SBATCH --output=/home/asidani/logs/clip_cycle/job_name_%j.log
#SBATCH --error=/home/asidani/logs/clip_cycle/job_name_%j.err


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

send_discord "[${SLURM_JOB_ID}]: Starting Training with Parallel Datasets"


TRAIN_PARAMS_PARALLEL=(
    --style_a en
    --style_b fr
    --lang en
    # Using parallel training datasets instead of mono
    --path_paral_A_train /home/asidani/TST-CycleGAN/data/multi30k/data/task1/raw/train.en
    --path_paral_B_train /home/asidani/TST-CycleGAN/data/multi30k/data/task1/raw/train.fr
    --path_paral_A_eval /home/asidani/TST-CycleGAN/data/multi30k/data/task1/raw/test_2016_val.en
    --path_paral_B_eval /home/asidani/TST-CycleGAN/data/multi30k/data/task1/raw/test_2016_val.fr
    --path_paral_eval_ref /home/asidani/TST-CycleGAN/data/multi30k/data/task1/raw/val_fr/
    --n_references 1
    --shuffle
    --generator_model_tag facebook/mbart-large-50
    --discriminator_model_tag distilbert-base-multilingual-cased
    # --pretrained_classifier_model ./classifiers/paradetox/roberta-base_10/
    --lambdas "1|1|1|1|0"
    # --from_pretrained ./ckpts_de_1e5_clip/epoch_24/
    --epochs 30
    --learning_rate 1e-5
    --max_sequence_length 64
    --batch_size 16
    --num_workers 1
    --save_base_folder ./ckpts_fr_1e5_clip_supervised_parallel/
    --save_steps 1
    --eval_strategy epochs
    --eval_steps 1
    --pin_memory
    --use_cuda_if_available
    --comet_logging
    --supervised_training
    --supervised_loss_weight 1
    # --comet_key ""
    # --comet_workspace ""
    # --comet_exp "c910b29ce9854515bf27119552481de9"
    # --comet_project_name ""
    --use_cliptrans
    --cliptrans_stage1_ckpt /home/asidani/CLIPTrans/models/
)
python train.py "${TRAIN_PARAMS_PARALLEL[@]}"



send_discord "[${SLURM_JOB_ID}]: Train completed"

LOG_FILE="/home/asidani/logs/clip_cycle/job_name_${SLURM_JOB_ID}.log"
ERR_FILE="/home/asidani/logs/clip_cycle/job_name_${SLURM_JOB_ID}.err"

python3 /home/asidani/notif.py "$LOG_FILE" "$ERR_FILE" 