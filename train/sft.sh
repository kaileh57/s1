#!/bin/bash
uid="$(date +%Y%m%d_%H%M%S)"
base_model="google/gemma-3-12b-it"
lr=1e-5
min_lr=0
epochs=3
weight_decay=1e-4
micro_batch_size=1
gradient_accumulation_steps=1
max_steps=-1
gpu_count=$(nvidia-smi -L | wc -l)
push_to_hub=true

torchrun --nproc-per-node ${gpu_count} --master_port 12345 \
    train/sft.py \
    --block_size=16384 \
    --per_device_train_batch_size=${micro_batch_size} \
    --per_device_eval_batch_size=${micro_batch_size} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --num_train_epochs=${epochs} \
    --train_file_path="Sculptor-AI/s1K-claude-gemma-tokenized" \
    --model_name=${base_model} \
    --warmup_ratio=0.05 \
    --fsdp="full_shard auto_wrap" \
    --fsdp_config="train/fsdp_config_gemma.json" \
    --bf16=True \
    --eval_strategy="steps" \
    --eval_steps=50 \
    --logging_steps=1 \
    --save_steps=100 \
    --lr_scheduler_type="cosine" \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --output_dir="ckpts/gemma-claude-s1-${uid}" \
    --hub_model_id="Sculptor-AI/gemma-3-12b-claude-s1" \
    --push_to_hub=${push_to_hub} \
    --save_only_model=True \
    --gradient_checkpointing=True