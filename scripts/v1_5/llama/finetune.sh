

phi2='microsoft/phi-2'
llama='meta-llama/Meta-Llama-3.1-8B-Instruct'

deepspeed llava/train/train_mem.py \
--deepspeed ./scripts/zero2.json \
--model_name_or_path $llama \
--version llama_3_1 \
--data_path ./playground/data/llava_v1_5_mix665k.json \
--image_folder ./playground/data \
--vision_tower openai/clip-vit-large-patch14-336 \
--llm_backbone llama_3_1 \
--llm_pad_token pad \
--tune_mm_mlp_adapter True \
--use_contrastive_loss False \
--clip_loss_coef 1 \
--mm_vision_select_layer -2 \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--image_aspect_ratio pad \
--group_by_modality_length True \
--bf16 True \
--output_dir ./ckpts/llama/llava-llama-test \
--num_experts_per_tok 2 \
--num_experts 4 \
--num_layers 2 \
--num_heads 2 \
--aux_loss_coef 0.01 \
--share_moe False \
--cross_attention False \
--mm_projector_type 'mlp2x_gelu' \
--num_train_epochs 1 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--gradient_accumulation_steps 1 \
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