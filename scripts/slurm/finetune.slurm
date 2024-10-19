#!/bin/bash -l

#SBATCH -A collabrobogroup
#SBATCH --array=0
#SBATCH --ntasks=1 
#SBATCH -t 12:00:00 
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
llama3_path=aimagelab/LLaVA_MORE-llama_3_1-8B-finetuning
images_path=/scratch/mi8uu/mrm/vqa_synthetic_data/LLaVA/resource/main/images
data_train_path=/scratch/mi8uu/mrm/vqa_synthetic_data/LLaVA/resource/main/train/dataset.json
vision_tower=openai/clip-vit-large-patch14-336

job_name="finetune-moe"
echo "job name: $job_name"

PORT=$((29500 + SLURM_JOB_ID % 1000))

apptainer exec --nv /scratch/mi8uu/mrm/sifs/llava_container_v2.sif deepspeed --master_port $PORT llava/train/train_mem.py \
--deepspeed ./scripts/zero3.json \
--llm_backbone llama_3 \
--llm_pad_token pad \
--model_name_or_path $llama3_path \
--version llama_3 \
--data_path $data_train_path \
--image_folder $images_path \
--vision_tower $vision_tower \
--mm_projector_type mlp2x_gelu \
--moe_enable True \
--num_experts 4 \
--num_experts_per_tok 2 \
--router_aux_loss_coef 0.01 \
--mm_vision_select_layer -2 \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--image_aspect_ratio pad \
--group_by_modality_length True \
--bf16 True \
--output_dir ./ckpts/${job_name} \
--num_train_epochs $epochs \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 2 \
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
--model_max_length 2048 \
--gradient_checkpointing True \
--dataloader_num_workers 8 \
--lazy_preprocess True \
--report_to wandb \
--run_name $job_name \