#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name="master-fafnir"
#SBATCH --time=10:00:00
#SBATCH --partition=GPUQ
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=4
# Increase this when you want more than one GPU for distributed training.
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --constraint="(a100)&(gpu80g)"
#SBATCH --output=slurm_outputs/output_master.txt
#SBATCH --error=slurm_outputs/output_master.err


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

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    OLD_IFS=$IFS
    IFS=','
    set -- $CUDA_VISIBLE_DEVICES
    IFS=$OLD_IFS
    NUM_PROCESSES=$#
else
    NUM_PROCESSES=${SLURM_GPUS_ON_NODE:-1}
fi

echo "Launching training with ${NUM_PROCESSES} process(es)"

if [ "${NUM_PROCESSES}" -gt 1 ]; then
    uv run accelerate launch \
        --multi_gpu \
        --num_processes "${NUM_PROCESSES}" \
        --num_machines 1 \
        train.py
else
    uv run accelerate launch --num_processes 1 train.py
fi

echo "Job finished"
