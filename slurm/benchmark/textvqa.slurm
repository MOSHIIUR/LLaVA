#!/bin/bash -l

#SBATCH -A collabrobogroup
#SBATCH --array=0
#SBATCH --ntasks=1 
#SBATCH -t 00:30:00 
#SBATCH -p gpu
#SBATCH --gres=gpu:a40:1
#SBATCH -N 1 
#SBATCH --cpus-per-task=32
#SBATCH --output=log/slurm/evaluation/textVQA/log-%A-%a.log 
#SBATCH -J benchmark

export TORCH_HOME=/project/CollabRoboGroup/.cache
export TRANSFORMERS_CACHE=/project/CollabRoboGroup/.cache


module purge
module load apptainer

model_paths=('./ckpts/llama/finetune/llava-llama-c01/' './ckpts_it/baseline/llava-base-v2' './ckpts_it/moe_full/llava-moe-e4t2-finetune' './ckpts_it/moe_full/llava-moe-e5t3-finetune' './ckpts_it/moe/llava-moe-e8t2-finetune')
model_name=('llava-llama-base' 'base-v2' 'moe-e4t4' 'moe-e5t3' 'moe-e8t2')

model_name=${file_names[${SLURM_ARRAY_TASK_ID}]}
model_path=${model_paths[${SLURM_ARRAY_TASK_ID}]}

apptainer exec --nv /scratch/mi8uu/mrm/sifs/llava_container_v2.sif \
python3 -m llava.eval.model_vqa_loader \
    --model-path ${model_path} \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/${model_name}/${model_name}.jsonl \
    --temperature 0 \
    --conv-mode llama_3_1

apptainer exec --nv /scratch/mi8uu/mrm/sifs/llava_container_v2.sif \
python3 -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/${model_name}/${model_name}.jsonl