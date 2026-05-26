#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name="benchmark-resolution"
#SBATCH --time=0-04:00:00
#SBATCH --partition=GPUQ
#SBATCH --mem=80GB
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --constraint="(a100)&(gpu80g)"
#SBATCH --output=slurm_outputs/benchmark_resolution.txt
#SBATCH --error=slurm_outputs/benchmark_resolution.err


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

echo "Benchmarking Fafnir-Mamba across resolutions..."
uv run python benchmark_resolution.py --config experiment_2.yaml

echo "Benchmarking Fafnir-DETR-encoder across resolutions..."
uv run python benchmark_resolution.py --config experiment_2_detr.yaml

echo "Rendering two-panel chart..."
uv run python plot_resolution_benchmark.py \
    --csvs results/benchmark/experiment_2.csv results/benchmark/experiment_2_detr.csv \
    --labels Fafnir-Mamba Fafnir-DETR \
    --output docs/images/resolution_benchmark.png
