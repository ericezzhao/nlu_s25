#!/bin/bash

# Please adjust these settings according to your needs.
#SBATCH --partition=gpu_misc_v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=truthfulqa
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ez2545@nyu.edu
#SBATCH --output="truthfulqa.out"


module purge
cd /scratch/ez2545/nlu_s25/hw3/hpc
module load cuda/11.6.2
source llava_env/bin/activate
echo "Checking installed packages..."
pip show transformers evaluate || pip install transformers evaluate
bash run_truthfulqa_evaluation.sh