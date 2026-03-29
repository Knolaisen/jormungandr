#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name="master-fafnir"
#SBATCH --time=03-12:00:00
#SBATCH --partition=GPUQ
#SBATCH --mem=80GB
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
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

# ── Accelerate environment variables ──
# Number of GPUs allocated by SLURM
export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "Number of GPUs: $NUM_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Run with accelerate launch instead of plain python
uv run accelerate launch \
    --num_processes=$NUM_GPUS \
    --mixed_precision=fp16 \
    train.py --config experiment_1.yaml

echo "Job finished"
 
 
