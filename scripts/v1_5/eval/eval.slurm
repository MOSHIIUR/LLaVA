#!/bin/bash -l

#SBATCH -A collabrobogroup
#SBATCH --array=0-4
#SBATCH --ntasks=1 
#SBATCH -t 08:00:00 
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1 
#SBATCH --mem=256G
#SBATCH --constraint=a100_80gb
#SBATCH --output=log/slurm/evaluation/log-%A-%a.log 
#SBATCH -J eval

export TORCH_HOME=/project/CollabRoboGroup/.cache
export TRANSFORMERS_CACHE=/project/CollabRoboGroup/.cache

model_names=('llava_base' 'llava_base_v2' 'llava_moe_e4t2' 'llava_moe_e5t3' 'llava_moe_e8t2')
models_phi=('llava_phi_c01' 'llava_phi_c02' 'llava_phi_c03' 'llava_phi_c04' 'llava_phi_c05' 'llava_phi_c06' 'llava_phi_c09' 'llava_phi_c10')
target_model=${model_names[${SLURM_ARRAY_TASK_ID}]}
target_model_phi=${models_phi[${SLURM_ARRAY_TASK_ID}]}


module purge
module load apptainer


apptainer exec --nv /scratch/mi8uu/mrm/sifs/llava_container_VLMEvalKit.sif python3 VLMEvalKit/run.py \
    --data ScienceQA_VAL TextVQA_VAL POPE MME Q-Bench1_VAL SEEDBench_IMG MMMU_DEV_VAL HallusionBench AI2D_TEST RealWorldQA \
    --model $target_model \
    --verbose
