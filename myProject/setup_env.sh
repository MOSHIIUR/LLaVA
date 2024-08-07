#!/bin/bash


# Store the current directory
previous_dir=$(pwd)

# Create conda environment and install packages
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation

# Set up environment variables using Python
python -c '
import os
import wandb
wandb.login(key="a9cd634aa6e8ac711e4a18b073f238ff96672289")
os.environ["WANDB_PROJECT"]="FineTuneLLaVa"
'


