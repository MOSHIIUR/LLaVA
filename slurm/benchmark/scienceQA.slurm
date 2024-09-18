#!/bin/bash -l

#SBATCH -A collabrobogroup
#SBATCH --array=1
#SBATCH --ntasks=1 
#SBATCH -t 00:30:00 
#SBATCH -p gpu
#SBATCH --gres=gpu:a40:1
#SBATCH -N 1 
#SBATCH --cpus-per-task=32
#SBATCH --output=log/slurm/evaluation/scienceQA/log-%A-%a.log 
#SBATCH -J SQA

export TORCH_HOME=/project/CollabRoboGroup/.cache
export TRANSFORMERS_CACHE=/project/CollabRoboGroup/.cache


module purge
module load apptainer

model_paths=('./ckpts/llama/finetune/llava-llama-c01/' './ckpts/llama/finetune/llava-more/' './ckpts_it/moe_full/llava-moe-e4t2-finetune' './ckpts_it/moe_full/llava-moe-e5t3-finetune' './ckpts_it/moe/llava-moe-e8t2-finetune')
file_names=('llama_base' 'llava-more' 'moe-e4t4' 'moe-e5t3' 'moe-e8t2')
outputs=('llama_base_output' 'llava-more_output' 'moe-e5t3_output' 'moe-e8t2_output')
results=('llama_base_result' 'llava-more_result' 'moe-e5t3_result' 'moe-e8t2_result')

current_file_name=${file_names[${SLURM_ARRAY_TASK_ID}]}
current_output=${outputs[${SLURM_ARRAY_TASK_ID}]}
current_result=${results[${SLURM_ARRAY_TASK_ID}]}
current_model_path=${model_paths[${SLURM_ARRAY_TASK_ID}]}

apptainer exec --nv /scratch/mi8uu/mrm/sifs/llava_container_v2.sif \
python3 -m llava.eval.model_vqa_science \
    --model-path ${current_model_path} \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/${current_file_name}/${current_file_name}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

apptainer exec --nv /scratch/mi8uu/mrm/sifs/llava_container_v2.sif \
python3 llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/${current_file_name}/${current_file_name}.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/${current_file_name}/output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/${current_file_name}/result.json
