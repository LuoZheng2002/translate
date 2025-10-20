#!/bin/bash
#SBATCH --job-name=bfcl_generate       # Job name
#SBATCH --output=bfcl_generate_%j.out        # Standard output log (%j = job ID)
#SBATCH --error=bfcl_generate_%j.err         # Error log
#SBATCH --time=00:10:00                 # Max runtime (HH:MM:SS)
# #SBATCH --partition=gpu                  # Partition/queue with GPU
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=luozheng@usc.edu
#SBATCH --gres=gpu:4                     # Request 4 GPU
#SBATCH --cpus-per-task=4                # CPU cores
#SBATCH --mem=32G                        # RAM
#SBATCH --account=bfdz-delta-gpu    # Your Slurm account/project

# Load modules or activate your environment
# module load python/3.12.1

# Go to your project directory
# cd /u/zluo8/qwen

source activate_environment.sh

# Set HF_HOME environment variable
export HF_HOME=/work/nvme/bfdz/zluo8/huggingface
# Convert the notebook to a .py script
# jupyter nbconvert --to script test_toolrl.ipynb

# Run the training script

# ibm-granite/granite-3.1-8b-instruct

bfcl generate --model ibm-granite/granite-3.1-8b-instruct --test-category multiple --backend vllm --result-dir /projects/bfdz/zluo8/translate/result --num-gpus 4 --gpu-memory-utilization 0.95
