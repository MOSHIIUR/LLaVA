#!/bin/bash -l

#SBATCH -A collabrobogroup
#SBATCH --array=10
#SBATCH --ntasks=1 
#SBATCH -t 00:30:00 
#SBATCH -p gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH -N 1 
#SBATCH --cpus-per-task=32
#SBATCH --output=log/slurm/evaluation/scienceQA/log-%A-%a.log 
#SBATCH -J SQA

export TORCH_HOME=/project/CollabRoboGroup/.cache
export TRANSFORMERS_CACHE=/project/CollabRoboGroup/.cache
export PYTHONPATH=/app:/opt/conda/lib/python3.10/site-packages:.


module purge
module load apptainer

# for new combination
# add model_paths, file_names, version, sif, count+=1
count=13

model_paths=('./ckpts_it/baseline/llava-base' './ckpts_it/baseline/llava-base-v2' './ckpts_it/moe_full/llava-moe-e4t2-finetune' './ckpts_it/moe_full/llava-moe-e5t3-finetune' './ckpts_it/moe/llava-moe-e8t2-finetune' './ckpts/llama/finetune/llava-llama-c01/' './ckpts/llama/finetune/llava-llama-c02' './ckpts/llama/finetune/llava-llama-c04' './ckpts/llama/finetune/llava-llama-c05' './ckpts/phi2/finetune/llava-phi-c01' './ckpts/phi2/finetune/llava-phi-c02' './ckpts/phi2/finetune/llava-phi-c05' './ckpts/phi2/finetune/llava-phi-c06')
file_names=('base' 'base-v2' 'moe-e4t4' 'moe-e5t3' 'moe-e8t2' 'llava-llama-c01' 'llava-llama-c02' 'llava-llama-c04' 'llava-llama-c05' 'llava-phi-c01' 'llava-phi-c02' 'llava-phi-c05' 'llava-phi-c06')
version=(vicuna_v1 vicuna_v1 vicuna_v1 vicuna_v1 vicuna_v1 llama_3_1 llama_3_1 llama_3_1 llama_3_1 phi phi phi phi)
sif=('llava' 'llava' 'llava' 'llava' 'llava' 'llava_container_v2' 'llava_container_v2' 'llava_container_v2' 'llava_container_v2' 'llava_container_phi' 'llava_container_phi' 'llava_container_phi' 'llava_container_phi')

current_file_name=${file_names[${SLURM_ARRAY_TASK_ID}]}
current_model_path=${model_paths[${SLURM_ARRAY_TASK_ID}]}
conv_version=${version[${SLURM_ARRAY_TASK_ID}]}
container=${sif[${SLURM_ARRAY_TASK_ID}]}


apptainer exec --nv /scratch/mi8uu/mrm/sifs/${container}.sif \
python3 -m llava.eval.model_vqa_science \
    --model-path ${current_model_path} \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/${current_file_name}/${current_file_name}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode $conv_version

apptainer exec --nv /scratch/mi8uu/mrm/sifs/${container}.sif \
python3 llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/${current_file_name}/${current_file_name}.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/${current_file_name}/output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/${current_file_name}/result.json
