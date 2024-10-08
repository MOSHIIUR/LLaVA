#!/bin/bash -l

#SBATCH -A collabrobogroup
#SBATCH --array=0-4 
#SBATCH --ntasks=1 
#SBATCH -t 01:00:00 
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:2
#SBATCH -N 1 
#SBATCH --cpus-per-task=64
#SBATCH --output=log/slurm/pretrain_wcattn/log-%A-%a.log 
#SBATCH -J wcattn

export TORCH_HOME=/project/CollabRoboGroup/.cache
export TRANSFORMERS_CACHE=/project/CollabRoboGroup/.cache

module purge
module load apptainer

model_name=('lmsys/vicuna-13b-v1.5' 'lmsys/vicuna-7b-v1.5' 'lmsys/vicuna-13b-v1.5' 'lmsys/vicuna-13b-v1.5' 'lmsys/vicuna-13b-v1.5')
vision_tower=('openai/clip-vit-large-patch14-336' 'openai/clip-vit-large-patch14-336' 'openai/clip-vit-large-patch14-336' 'openai/clip-vit-large-patch14-336' 'openai/clip-vit-large-patch14-336')
num_experts=(3 4 4 5 6)
num_experts_per_tok=(2 2 3 2 3)
num_layers=(12 14 16 18 24)
num_heads=(16 16 16 16 16)
batch_size=(32 32 32 32 32)
output_dir=('./ckpts/pretrain_wcattn/llava-v1.5-13b-pretrain-v1' './ckpts/pretrain_wcattn/llava-v1.5-13b-pretrain-v2' './ckpts/pretrain_wcattn/llava-v1.5-13b-pretrain-v3' './ckpts/pretrain_wcattn/llava-v1.5-13b-pretrain-v4' './ckpts/pretrain_wcattn/llava-v1.5-13b-pretrain-v5')
PORT=$((29500 + SLURM_JOB_ID % 1000))


apptainer exec --nv /scratch/mi8uu/mrm/llava.sif deepspeed --master_port $PORT llava/train/train_mem.py \
--deepspeed ./scripts/zero2.json \
--model_name_or_path ${model_name[${SLURM_ARRAY_TASK_ID}]} \
--version plain \
--data_path ./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
--image_folder ./playground/data/images \
--vision_tower openai/clip-vit-large-patch14-336 \
--tune_mm_mlp_adapter True \
--mm_vision_select_layer -2 \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--bf16 True \
--output_dir ${output_dir[${SLURM_ARRAY_TASK_ID}]} \
--num_experts_per_tok ${num_experts_per_tok[${SLURM_ARRAY_TASK_ID}]} \
--num_experts ${num_experts[${SLURM_ARRAY_TASK_ID}]} \
--num_layers ${num_layers[${SLURM_ARRAY_TASK_ID}]} \
--num_heads ${num_heads[${SLURM_ARRAY_TASK_ID}]} \
--aux_loss_coef 0.01 \
--share_moe False \
--cross_attention False \
--mm_projector_type 'mlp2x_gelu' \
--num_train_epochs 1 \
--per_device_train_batch_size ${batch_size[${SLURM_ARRAY_TASK_ID}]} \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 2500 \
--save_total_limit 1 \
--learning_rate 1e-3 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--model_max_length 2048 \
--gradient_checkpointing True \
--dataloader_num_workers 4 \
--lazy_preprocess True \
--report_to wandb \
--run_name "UVA-Pretrain-mlp"