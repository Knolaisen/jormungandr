#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name="experiment-1"
#SBATCH --time=1:00:00
#SBATCH --partition=GPUQ
#SBATCH --mem=80GB
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --constraint="(a100)&(gpu80g)"
#SBATCH --output=slurm_outputs/output_experiment_1.txt
#SBATCH --error=slurm_outputs/output_experiment_1.err


WORKDIR=../${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"


module purge
module load Anaconda3/2024.02-1

pip install uv --user

uv sync

# Run the Python script
uv run python train.py --config experiment_1.yaml


echo "Job finished"
