#!/bin/bash -l

#SBATCH -A collabrobogroup
#SBATCH --array=7
#SBATCH --ntasks=1 
#SBATCH -t 26:00:00 
#SBATCH -p gpu
#SBATCH --gres=gpu:a6000:4
#SBATCH -N 1 
#SBATCH --cpus-per-task=16
#SBATCH --exclude=udc-an28-[1,7]
#SBATCH --output=log/slurm/phi_finetune/log-%A-%a.log 
#SBATCH -J phi_finetune

export TORCH_HOME=/project/CollabRoboGroup/.cache
export TRANSFORMERS_CACHE=/project/CollabRoboGroup/.cache

module purge
module load apptainer

num_experts=(4 4 4 4 4 4 4 4 4 4)
num_experts_per_tok=(2 2 2 2 2 2 2 2 2 2)
num_layers=(2 2 2 2 2 2 2 2 2)
num_heads=(2 2 2 2 2 2 2 2 2)
batch_size=(16 16 16 16 8 8 8 8 8 8 8 8 8)
pretrain_embed_tokens=(True True True True True True False)
share_moe=(False False True True False False True True)
use_clip_loss=(True True True False True False False True)
cross_attention=(False False False False True True True True)
mm_projector_type=('mlp2x_gelu' 'sparse_moe' 'sparse_moe' 'sparse_moe' 'mlp2x_gelu' 'sparse_moe' 'sparse_moe' 'sparse_moe')
clip_loss_coef=(0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05)

PORT=$((29500 + SLURM_JOB_ID % 1000))

BOD='./ckpts/phi2/finetune'
output_dir=( $BOD/llava-phi-c01 $BOD/llava-phi-c02 $BOD/llava-phi-c03 $BOD/llava-phi-c04 $BOD/llava-phi-c05 $BOD/llava-phi-c06 $BOD/llava-phi-c07 $BOD/llava-phi-c08)


phi2='microsoft/phi-2'

apptainer exec --nv /scratch/mi8uu/mrm/sifs/llava_container_phi.sif deepspeed llava/train/train_mem.py \
--deepspeed ./scripts/zero2.json \
--model_name_or_path $phi2 \
--version phi \
--data_path ./playground/data/llava_v1_5_mix665k.json \
--image_folder ./playground/data \
--vision_tower openai/clip-vit-large-patch14-336 \
--pretrain_mm_mlp_adapter './ckpts/phi2/pretrain/llava-phi-FF/mm_projector.bin' \
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
--gradient_accumulation_steps 2 \
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
--model_max_length 2048 \
--gradient_checkpointing True \
--dataloader_num_workers 4 \
--lazy_preprocess True \
--report_to wandb \
--run_name "UVA-Pretrain-mlp"