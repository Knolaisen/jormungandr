#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name="calculate-params-and-flops"
#SBATCH --time=1-00:00:00
#SBATCH --partition=GPUQ
#SBATCH --mem=80GB
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --constraint="(a100)&(gpu80g)"
#SBATCH --output=slurm_outputs/calculate_params_and_flops.txt
#SBATCH --error=slurm_outputs/calculate_params_and_flops.err


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
echo "Calculating parameters and FLOPs for the experiments with different numbers of Encoder layers..."
uv run python calculate_gflops.py --config experiment_2_3_layers.yaml
uv run python calculate_gflops.py --config experiment_2_6_layers.yaml
uv run python calculate_gflops.py --config experiment_2_24_layers.yaml

echo "Calculating parameters and FLOPs for the experiments with different numbers of State sizes ..."
uv run python calculate_gflops.py --config experiment_2_hidden_dim_16.yaml
uv run python calculate_gflops.py --config experiment_2_hidden_dim_32.yaml
uv run python calculate_gflops.py --config experiment_2_hidden_dim_64.yaml
uv run python calculate_gflops.py --config experiment_2_hidden_dim_128.yaml

echo "Calculating parameters and FLOPs for the experiments with different Mamba types..."
uv run python calculate_gflops.py --config experiment_2_mamba1.yaml

echo "Calculating parameters and FLOPs for the experiments with different Encoder layers FFN"
uv run python calculate_gflops.py --config experiment_2_ffn.yaml
uv run python calculate_gflops.py --config experiment_2_ffn_265_hidden_size.yaml

echo "Calculating parameters and FLOPs for the experiments with DETR..."
uv run python calculate_gflops.py --config experiment_2_detr.yaml

echo "Calculating parameters and FLOPs for the experiments with different numbers of Encoder layers and State sizes..."
uv run python calculate_gflops.py --config best_fafnir_12_enc.yaml


echo "Calculating parameters and FLOPs for the experiments with Jormungandr..."
uv run python calculate_gflops.py --config experiment_5_2.yaml
uv run python calculate_gflops.py --config experiment_5_4.yaml
uv run python calculate_gflops.py --config experiment_5_8.yaml
uv run python calculate_gflops.py --config experiment_5_64.yaml


