#!/bin/bash -l

#SBATCH -A collabrobogroup
#SBATCH --array=2 # 2,5-9 -> a100 2 -> a6000
#SBATCH --ntasks=1 
#SBATCH -t 30:00:00 
#SBATCH -p gpu
#SBATCH --gres=gpu:a6000:8
#SBATCH --cpus-per-task=40
#SBATCH --mem=128G
#SBATCH -N 1 
#SBATCH --output=log/slurm/llama/log-%A-%a.log 
#SBATCH -J llama

export TORCH_HOME=/project/CollabRoboGroup/.cache
export TRANSFORMERS_CACHE=/project/CollabRoboGroup/.cache

module purge
module load apptainer

num_experts=(4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4)
num_experts_per_tok=(2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2)
num_layers=(2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2)
num_heads=(2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2)
batch_size=(16 8 16 16 16 8 8 8 8 8)
grad_accumulation=(1 2 1 1 1 2 2 2 2 2)
pretrain_embed_tokens=(False False False False True True False False False False False True True False)
share_moe=(False False True False False True True False False True)
use_clip_loss=(False False False True True True False True False False)
cross_attention=(False False False False False False False True True True)
mm_projector_type=('mlp2x_gelu' 'sparse_moe' 'sparse_moe' 'mlp2x_gelu' 'sparse_moe' 'sparse_moe' 'sparse_moe' 'mlp2x_gelu' 'sparse_moe' 'sparse_moe')
clip_loss_coef=(0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05)

PORT=$((29500 + SLURM_JOB_ID % 1000))

BOD='./ckpts/llama/finetune'
output_dir=( $BOD/llava-llama-c01 $BOD/llava-llama-c02 $BOD/llava-llama-c03 $BOD/llava-llama-c04 $BOD/llava-llama-c05 $BOD/llava-llama-c06 $BOD/llava-llama-c07 $BOD/llava-llama-c08 $BOD/llava-llama-c09 $BOD/llava-llama-c10 $BOD/llava-llama-c11)

llama='meta-llama/Meta-Llama-3.1-8B-Instruct'

apptainer exec --nv /scratch/mi8uu/mrm/sifs/llava_container_v2.sif deepspeed --master_port $PORT llava/train/train_mem.py \
--deepspeed ./scripts/zero3.json \
--model_name_or_path $llama \
--version llama_3_1 \
--data_path ./playground/data/llava_v1_5_mix665k.json \
--image_folder ./playground/data \
--vision_tower openai/clip-vit-large-patch14-336 \
--llm_backbone llama_3_1 \
--llm_pad_token pad \
--pretrain_mm_mlp_adapter './pretrained_weight/LLaVA_MORE-llama_3_1-8B-pretrain/mm_projector.bin' \
--use_contrastive_loss ${use_clip_loss[${SLURM_ARRAY_TASK_ID}]} \
--clip_loss_coef ${clip_loss_coef[${SLURM_ARRAY_TASK_ID}]} \
--mm_vision_select_layer -2 \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--image_aspect_ratio pad \
--group_by_modality_length True \
--bf16 True \
--output_dir ${output_dir[${SLURM_ARRAY_TASK_ID}]} \
--num_experts_per_tok ${num_experts_per_tok[${SLURM_ARRAY_TASK_ID}]} \
--num_experts ${num_experts[${SLURM_ARRAY_TASK_ID}]} \
--num_layers ${num_layers[${SLURM_ARRAY_TASK_ID}]} \
--num_heads ${num_heads[${SLURM_ARRAY_TASK_ID}]} \
--aux_loss_coef 0.01 \
--share_moe ${share_moe[${SLURM_ARRAY_TASK_ID}]} \
--cross_attention ${cross_attention[${SLURM_ARRAY_TASK_ID}]} \
--mm_projector_type ${mm_projector_type[${SLURM_ARRAY_TASK_ID}]} \
--num_train_epochs 1 \
--per_device_train_batch_size ${batch_size[${SLURM_ARRAY_TASK_ID}]} \
--per_device_eval_batch_size 16 \
--gradient_accumulation_steps ${grad_accumulation[${SLURM_ARRAY_TASK_ID}]} \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 50000 \
--save_total_limit 1 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--model_max_length 4096 \
--gradient_checkpointing True \
--dataloader_num_workers 4 \
--lazy_preprocess True \
--report_to wandb \
--run_name "UVA-Pretrain-mlp"