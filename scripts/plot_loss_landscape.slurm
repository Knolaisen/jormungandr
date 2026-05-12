#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name="plot-loss-landscape"
#SBATCH --time=04:00:00
#SBATCH --partition=GPUQ
#SBATCH --mem=80GB
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --constraint="(a100)&(gpu80g)"
#SBATCH --output=slurm_outputs/output_plot_loss_landscape.txt
#SBATCH --error=slurm_outputs/output_plot_loss_landscape.err


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

# Plot the Fafnir loss landscape for run igeia7g6 (artifact mamba_giouloss_20260505_023926:v2).
# Trained from experiment_3_0.yaml; eval done without autograd so a larger batch fits.
uv run python plot_loss_landscape.py \
    --config-name experiment_3_0.yaml \
    --wandb-artifact sverrenystad-ntnu/Fafnir/mamba_giouloss_20260505_023926:v2 \
    --grid-size 41 \
    --extent 1.0 \
    --subset-size 256 \
    --batch-size 16 \
    --seed 42


echo "Job finished"
