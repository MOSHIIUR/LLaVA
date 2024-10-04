#!/bin/bash -l

#SBATCH -A collabrobogroup
#SBATCH --array=0
#SBATCH --ntasks=1 
#SBATCH -t 01:00:00 
#SBATCH -p gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH -N 1 
#SBATCH --mem=128G
#SBATCH --output=log/slurm/evaluation/log-%A-%a.log 
#SBATCH -J eval

export TORCH_HOME=/project/CollabRoboGroup/.cache
export TRANSFORMERS_CACHE=/project/CollabRoboGroup/.cache

module purge
module load apptainer


apptainer exec --nv /scratch/mi8uu/mrm/sifs/llava_container_eval.sif python3 -m accelerate.commands.launch \
    --num_processes=8 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="./ckpts_it/baseline/llava-base" \
    --tasks mme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava-base_mme \
    --output_path ./logs_eval/