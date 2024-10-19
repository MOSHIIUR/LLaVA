#!/bin/bash

export WANDB_PROJECT=FineTuneLLaVa
export TOKENIZER_PATH=meta-llama/Meta-Llama-3.1-8B-Instruct
export WANDB_API_KEY=
export HF_TOKEN=

IFS=',' read -r -a nodelist <<<$SLURM_NODELIST
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=`comm -23 <(seq 5000 6000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
export OMP_NUM_THREADS=1

echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $SLURM_GPUS_PER_NODE"
echo "MASTER ADDR: ${MASTER_ADDR}"
echo "MASTER PORT: ${MASTER_PORT}"

epochs=1
llama3_path=meta-llama/Meta-Llama-3.1-8B-Instruct
images_path=playground/data/LLaVA-Pretrain/images
data_train_path=playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json
vision_tower=openai/clip-vit-large-patch14-336

job_name="llava-llama-llm-moe"
echo "job name: $job_name"

deepspeed llava/train/train_mem.py \
--deepspeed ./scripts/zero2.json \
--llm_backbone llama_3 \
--llm_pad_token pad \
--model_name_or_path $llama3_path \
--version plain \
--data_path $data_train_path \
--image_folder $images_path \
--vision_tower $vision_tower \
--mm_projector_type mlp2x_gelu \
--moe_enable True \
--num_experts 4 \
--num_experts_per_tok 2 \
--router_aux_loss_coef 0.01 \
--tune_mm_mlp_adapter True \
--mm_vision_select_layer -2 \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--bf16 True \
--output_dir ckpts/${job_name} \
--num_train_epochs $epochs \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 2 \
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
--model_max_length 2048 \
--gradient_checkpointing True \
--dataloader_num_workers 8 \
--lazy_preprocess True \
--report_to wandb \
--run_name $job_name \
