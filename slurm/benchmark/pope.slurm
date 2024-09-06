#!/bin/bash -l

#SBATCH -A collabrobogroup
#SBATCH --array=0-4
#SBATCH --ntasks=1 
#SBATCH -t 00:10:00 
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1 
#SBATCH --cpus-per-task=32
#SBATCH --output=log/slurm/evaluation/pope/log-%A-%a.log 
#SBATCH -J benchmark

export TORCH_HOME=/project/CollabRoboGroup/.cache
export TRANSFORMERS_CACHE=/project/CollabRoboGroup/.cache


module purge
module load apptainer

model_paths=('./ckpts_it/baseline/llava-base' './ckpts_it/baseline/llava-base-v2' './ckpts_it/moe_full/llava-moe-e4t2-finetune' './ckpts_it/moe_full/llava-moe-e5t3-finetune' './ckpts_it/moe/llava-moe-e8t2-finetune')
file_names=('base' 'base-v2' 'moe-e4t4' 'moe-e5t3' 'moe-e8t2')

model_name=${file_names[${SLURM_ARRAY_TASK_ID}]}
model_path=${model_paths[${SLURM_ARRAY_TASK_ID}]}

python3 -m llava.eval.model_vqa_loader \
    --model-path ${model_path} \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/{model_name}/answers.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python3 llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/{model_name}/answers.jsonl
