#!/bin/bash -l

#SBATCH -A collabrobogroup
#SBATCH --array=0
#SBATCH --ntasks=1 
#SBATCH -t 12:00:00 
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1 
#SBATCH --mem=256G
#SBATCH --constraint=a100_80gb
#SBATCH --output=log/slurm/evaluation/log-%A-%a.log 
#SBATCH -J eval

export TORCH_HOME=/project/CollabRoboGroup/.cache
export TRANSFORMERS_CACHE=/project/CollabRoboGroup/.cache



module purge
module load apptainer


apptainer exec --nv /scratch/mi8uu/mrm/sifs/llava_container_VLMEvalKit.sif python3 VLMEvalKit/run.py \
    --data ScienceQA_VAL TextVQA_VAL A-OKVQA POPE MME MMBench_TEST__EN MMBench_TEST__CN MMVet SEEDBench_IMG MMMU_DEV_VAL MathVista_MINI \
    --model llava_base \
    --verbose
