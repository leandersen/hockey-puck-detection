#!/bin/bash
#SBATCH --job-name=hockey-puck
#SBATCH --output=results/logs/%j.out
#SBATCH --error=results/logs/%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=shakeri_d

# Load modules
module load miniforge/24.11.3-py3.12

# Activate conda environment
conda activate hockey

# Move to project directory
cd $SLURM_SUBMIT_DIR

# Train
python src/train.py --config configs/train_config.yaml
