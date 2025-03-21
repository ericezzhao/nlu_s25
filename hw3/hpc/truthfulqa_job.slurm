#!/bin/bash

# Please adjust these settings according to your needs.
#SBATCH --account=ds_ga_1012-2024sp
#SBATCH --partition=gpu_misc_v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=truthfulqa
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ez2545@nyu.edu
#SBATCH --output="truthfulqa.out"


module purge
cd /scratch/ez2545/nlu_s25/hw3
bash run_truthfulqa_evaluation.sh
# OVERLAY_FILE=path/to/your/singularity/overlay/file
# SINGULARITY_IMAGE=path/to/your/singularity/image/file
# singularity exec --nv \
	    # --overlay $OVERLAY_FILE $SINGULARITY_IMAGE \
	    # /bin/bash -c "source /ext3/env.sh; bash hpc/run_truthfulqa_evaluation.sh"