#!/bin/bash -l

#SBATCH -A collabrobogroup
#SBATCH --array=0
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:00:00 
#SBATCH -p gpu
#SBATCH --nodes=4
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=32
#SBATCH --output=log/slurm/phi_pretrain/log-%A-%a.log 
#SBATCH -J phi_pretrain

export TORCH_HOME=/project/CollabRoboGroup/.cache
export TRANSFORMERS_CACHE=/project/CollabRoboGroup/.cache

module purge
module load apptainer

num_layers=(2 2 2 2 2 2 2 2 2 2 2)
num_heads=(2 2 2 2 2 2 2 2 2 2 2)
batch_size=(32 32 32 32 32 32 32 32 32 32 32)

# base output dir - BOD
BOD='./ckpts/phi2/pretrain'
output_dir=('$BOD/llava-phi-TT-0.01-test' '$BOD/llava-phi-TF-0.01' '$BOD/llava-phi-TT-1' '$BOD/llava-phi-TF-1' '$BOD/llava-phi-TT-0.05' '$BOD/llava-phi-TF-0.05' '$BOD/llava-phi-TT-0.1' '$BOD/llava-phi-TF-0.1' '$BOD/llava-phi-TT-0.5' '$BOD/llava-phi-TF-0.5' '$BOD/llava-phi-FF')


use_clip_loss=(True True True True True True True True True True False)
tune_embed_tokens=(True False True False True False True False True False False)
clip_loss_coef=(0.01 0.01 1 1 0.05 0.05 0.1 0.1 0.5 0.5 1)

PORT=$((29500 + SLURM_JOB_ID % 1000))

phi3='microsoft/Phi-3.5-mini-instruct'
phi2='microsoft/phi-2'

apptainer exec --nv /scratch/mi8uu/mrm/sifs/llava.sif deepspeed --master_port $PORT llava/train/train_mem.py \
--deepspeed ./scripts/zero2.json \
--model_name_or_path $phi2 \
--version plain \
--data_path ./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
--image_folder ./playground/data/images \
--vision_tower openai/clip-vit-large-patch14-336 \
--tune_mm_mlp_adapter True \
--tune_embed_tokens ${tune_embed_tokens[${SLURM_ARRAY_TASK_ID}]} \
--use_contrastive_loss ${use_clip_loss[${SLURM_ARRAY_TASK_ID}]} \
--clip_loss_coef ${clip_loss_coef[${SLURM_ARRAY_TASK_ID}]} \
--mm_vision_select_layer -2 \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--bf16 True \
--output_dir ${output_dir[${SLURM_ARRAY_TASK_ID}]} \
--num_layers ${num_layers[${SLURM_ARRAY_TASK_ID}]} \
--num_heads ${num_heads[${SLURM_ARRAY_TASK_ID}]} \
--cross_attention False \
--mm_projector_type mlp2x_gelu \
--num_train_epochs 1 \
--per_device_train_batch_size ${batch_size[${SLURM_ARRAY_TASK_ID}]} \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 2500 \
--save_total_limit 1 \
--learning_rate 2e-3 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--model_max_length 2048 \
--gradient_checkpointing True \
--dataloader_num_workers 4 \
--lazy_preprocess True \
--report_to wandb