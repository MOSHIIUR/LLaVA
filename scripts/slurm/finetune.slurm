#!/bin/bash -l

#SBATCH -A collabrobogroup
#SBATCH --array=0
#SBATCH --ntasks=1 
#SBATCH -t 01:00:00 
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:4
#SBATCH -N 1 
#SBATCH --mem=256G
#SBATCH --constraint=a100_80gb
#SBATCH --output=log/slurm/llama/log-%A-%a.log 
#SBATCH -J moe-llama

export TORCH_HOME=/project/CollabRoboGroup/.cache
export TRANSFORMERS_CACHE=/project/CollabRoboGroup/.cache
export WANDB_PROJECT=FineTuneLLaVa
export TOKENIZER_PATH=aimagelab/LLaVA_MORE-llama_3_1-8B-finetuning


module purge
module load apptainer

epochs=1
llama3_path=meta-llama/Meta-Llama-3.1-8B-Instruct
images_path=/scratch/mi8uu/mrm/LLaVA/playground/data/moe_llava
data_train_path=/scratch/mi8uu/mrm/LLaVA/playground/data/moe_llava/train_json
vision_tower=openai/clip-vit-large-patch14-336
mm_projector_path=LLaVA_MORE-llama_3_1-8B-pretrain

job_name="finetune-moe-llava-v01"
echo "job name: $job_name"

PORT=$((29500 + SLURM_JOB_ID % 1000))

apptainer exec --nv /scratch/mi8uu/mrm/sifs/llava_container_v2.sif deepspeed --master_port $PORT llava/train/train_mem.py \
--deepspeed ./scripts/zero3.json \
--model_name_or_path $llama3_path \
--llm_backbone llama_3_1 \
--llm_pad_token pad \
--version llama_3_1 \
--data_path ${data_train_path}/la_tune_256k.json \
            ${data_train_path}/lrv_tune_331k.json ${data_train_path}/lvis_tune_220k_.json \
            ${data_train_path}/svit_tune_157k.json ${data_train_path}/nlp_tune.json \
--image_folder $images_path \
--vision_tower $vision_tower \
--pretrain_mm_mlp_adapter $mm_projector_path \
--mm_projector_type mlp2x_gelu \
--mm_vision_select_layer -2 \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--image_aspect_ratio pad \
--group_by_modality_length True \
--bf16 True \
--output_dir ckpts/${job_name} \
--max_steps 10 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 1 \
--evaluation_strategy no \
--save_strategy steps \
--save_steps 24000 \
--save_total_limit 2 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type cosine \
--logging_steps 1 \
--tf32 True \
--model_max_length 4096 \
--gradient_checkpointing True \
--dataloader_num_workers 8 \
--lazy_preprocess True \
--report_to wandb \
--run_name $job_name \