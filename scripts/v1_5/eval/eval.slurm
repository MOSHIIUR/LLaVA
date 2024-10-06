#!/bin/bash -l

#SBATCH -A collabrobogroup
#SBATCH --array=0
#SBATCH --ntasks=1 
#SBATCH -t 01:00:00 
#SBATCH -p gpu
#SBATCH --gres=gpu:a6000:4
#SBATCH -N 1 
#SBATCH --mem=128G
#SBATCH --output=log/slurm/evaluation/log-%A-%a.log 
#SBATCH -J eval

export TORCH_HOME=/project/CollabRoboGroup/.cache
export TRANSFORMERS_CACHE=/project/CollabRoboGroup/.cache
export CUDA_VISIBLE_DEVICES=0,1,2,3 




module purge
module load apptainer


apptainer exec --nv /scratch/mi8uu/mrm/sifs/llava_container_VLMEvalKit.sif python3 VLMEvalKit/run.py \
    --data MMBench_DEV_EN MME SEEDBench_IMG \
    --model llava_base \
    --verbose
