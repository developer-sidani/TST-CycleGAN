#!/bin/bash
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --partition=cuda
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_cyclegan
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

source ~/.bashrc

source activate /home/asidani/.conda/envs/cyclegan

send_discord "[${SLURM_JOB_ID}]: Starting test"

/home/asidani/.conda/envs/cyclegan/bin/python3 train.py --style_a=en --style_b=de --lang=en \
                       --path_mono_A=/content/TST-CycleGAN/data/multi30k/text/data/task1/raw/train.en --path_mono_B=/content/TST-CycleGAN/data/multi30k/text/data/task1/raw/train.de \
                       --path_paral_A_eval=/content/TST-CycleGAN/data/multi30k/text/data/task1/raw/test_2016_val.en --path_paral_B_eval=/content/TST-CycleGAN/data/multi30k/text/data/task1/raw/test_2016_val.de --path_paral_eval_ref=/content/TST-CycleGAN/data/multi30k/text/data/task1/raw/data_flickr16_de--n_references=1 --shuffle \
                       --generator_model_tag=facebook/mbart-large-50 --discriminator_model_tag=distilbert-base-multilingual-cased \
                       --lambdas="10|1|1|1|0" --epochs=30 --learning_rate=5e-5 --max_sequence_length=64 --batch_size=64  \
                       --save_base_folder=./ckpts/ --save_steps=1 --eval_strategy=epochs --eval_steps=1  --pin_memory --use_cuda_if_available

                       send_discord "[${SLURM_JOB_ID}]: Test completed"

LOG_FILE="/home/asidani/logs/cyclegan/job_name_${SLURM_JOB_ID}.log"
ERR_FILE="/home/asidani/logs/cyclegan/job_name_${SLURM_JOB_ID}.err"

python3 /home/asidani/notif.py "$LOG_FILE" "$ERR_FILE"