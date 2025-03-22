#!/bin/bash

#SBATCH --job-name=truthfulqa
#SBATCH --account=ds_ga_1012-2025sp
#SBATCH --partition=n1s8-v100-1
#SBATCH --gres=gpu:v100:1
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --requeue
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ez2545@nyu.edu

module purge

OVERLAY_FILE=/scratch/$USER/pytorch-env/overlay-15GB-500K.ext3
SIF_IMAGE=/scratch/$USER/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

singularity exec --nv --overlay $OVERLAY_FILE:ro $SIF_IMAGE /bin/bash -c "
source /ext3/env.sh
cd /scratch/ez2545/nlu_s25/hw3
bash run_truthfulqa_evaluation.sh
"