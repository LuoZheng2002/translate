#!/bin/bash
#SBATCH --job-name=bfcl_evaluate       # Job name
#SBATCH --output=bfcl_evaluate_%j.out        # Standard output log (%j = job ID)
#SBATCH --error=bfcl_evaluate_%j.err         # Error log
#SBATCH --time=00:15:00                 # Max runtime (HH:MM:SS)
#SBATCH --partition=cpu                  # Partition/queue with GPU
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=luozheng@usc.edu
# #SBATCH --gres=gpu:4                     # Request 4 GPUs
#SBATCH --cpus-per-task=4                # CPU cores
#SBATCH --mem=32G                        # RAM
#SBATCH --account=bfdz-delta-gpu    # Your Slurm account/project

# Load modules or activate your environment
# module load python/3.12.1

# Go to your project directory
# cd /u/zluo8/qwen

source activate_environment.sh

# Convert the notebook to a .py script
# jupyter nbconvert --to script test_toolrl.ipynb

# Run the training script

# ibm-granite/granite-3.1-8b-instruct

bfcl evaluate --model ibm-granite/granite-3.1-8b-instruct --test-category multiple --result-dir /projects/bfdz/zluo8/translate/result  --score-dir /projects/bfdz/zluo8/translate/score

# bfcl evaluate --model DeepSeek-V3-0324-FC --test-category multiple --result-dir /projects/bfdz/zluo8/translate/result  --score-dir /projects/bfdz/zluo8/translate/score

# bfcl evaluate --model claude-sonnet-4-20250514 --test-category multiple --result-dir /projects/bfdz/zluo8/translate/result  --score-dir /projects/bfdz/zluo8/translate/score

# bfcl evaluate --model claude-3-5-haiku-20241022 --test-category multiple --result-dir /projects/bfdz/zluo8/translate/result  --score-dir /projects/bfdz/zluo8/translate/score


# bfcl evaluate --model gpt-4o-mini-2024-07-18 --test-category multiple --result-dir /projects/bfdz/zluo8/translate/result  --score-dir /projects/bfdz/zluo8/translate/score