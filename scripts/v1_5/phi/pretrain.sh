#!/bin/bash

llama='meta-llama/Meta-Llama-3.1-8B'
phi='microsoft/Phi-3.5-mini-instruct'
phi2='microsoft/phi-2'

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $phi2 \
    --llm_backbone phi2 \
    --version plain \
    --data_path ./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder ./playground/data/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --tune_mm_mlp_adapter True \
    --tune_embed_tokens False \
    --use_contrastive_loss False \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./ckpts/llava-llama-8b \
    --num_train_epochs 1 \
    --num_experts_per_tok 2 \
    --num_experts 4 \
    --num_layers 2 \
    --num_heads 2 \
    --clip_loss_coef 0.01 \
    --share_moe False \
    --cross_attention False \
    --mm_projector_type 'mlp2x_gelu'\
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
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
    --report_to wandb \
    --run_name "UVA-Pretrain-lora"
