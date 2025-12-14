#!/bin/bash
#SBATCH --job-name=train_NLP_model
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:a100
#SBATCH --mem=64G
#SBATCH -A msml612-fa25-class
#SBATCH --output=/home/suryajk/scratch.hpcintro/train_transformer_h_files/recent.out
#SBATCH --error=/home/suryajk/scratch.hpcintro/train_transformer_h_files/recent.err

# Load required modules
module load gcc/11.3.0
module load cuda/12.3.0

cd /home/suryajk/NLP

unset PYTHONPATH
source /home/suryajk/scratch.hpcintro/train_transformer_h_venv/bin/activate

export PYTHONPATH=/home/suryajk/scratch.hpcintro/train_transformer_h_venv/lib/python3.10/site-packages:$PYTHONPATH

/home/suryajk/scratch.hpcintro/train_transformer_h_venv/bin/accelerate launch \
	--num_processes 1 \
	--num_machines 1 \
	--mixed_precision fp16 \
	main.py
