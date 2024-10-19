#!/bin/bash -l

#SBATCH -A collabrobogroup
#SBATCH --array=0
#SBATCH --ntasks=1 
#SBATCH -t 12:00:00 
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:4
#SBATCH -N 1 
#SBATCH --cpus-per-task=64
#SBATCH --output=log/slurm/tune_embed/log-%A-%a.log 
#SBATCH -J tune_embed

export TORCH_HOME=/project/CollabRoboGroup/.cache
export TRANSFORMERS_CACHE=/project/CollabRoboGroup/.cache

module purge
module load apptainer

deepspeed llava/train/train_mem.py \
--deepspeed ./scripts/zero2.json \
--model_name_or_path $llama3_path \
--llm_backbone llama_3_1 \
--llm_pad_token pad \
--version plain \
--data_path $data_train_path \
--image_folder $images_path \
--vision_tower $vision_tower \
--mm_projector_type mlp2x_gelu \
--tune_mm_mlp_adapter True \
--mm_vision_select_layer -2 \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--bf16 True \
--output_dir ./checkpoints/${job_name} \
--num_train_epochs $epochs \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 1 \
--evaluation_strategy no \
--save_strategy steps \
--save_steps 24000 \
--save_total_limit 2 \
--learning_rate 1e-3 \
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