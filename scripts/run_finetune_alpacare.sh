#!/usr/bin/env bash

SEED=998244353
CHECKPOINTS_DIR="${HOME}/working/storage/checkpoints"
DATASET="lavita/AlpaCare-MedInstruct-52k"
DTYPE="bfloat16"
TRAIN_STEPS=6000
TRAIN_BATCH_SIZE=8
EVAL_BATCH_SIZE=8
GRADIENT_ACCUM_STEPS=8
NUM_GPUS=4

torchrun --standalone --nproc-per-node "$NUM_GPUS" run_finetune.py \
    --do_train \
    --checkpoints_dir "$CHECKPOINTS_DIR" \
    --seed "$SEED" \
    --dataset_path "$DATASET" \
    --input_field input \
    --output_field output \
    --instruction_field instruction \
    --prompt_template alpaca \
    --validation_size 1000 \
    --dataset_num_procs 4 \
    --model_checkpoint meta-llama/Llama-2-7b-chat-hf \
    --tokenizer_checkpoint meta-llama/Llama-2-7b-chat-hf \
    --max_seq_length 1024 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_r 16 \
    --lora_target_modules q_proj v_proj k_proj o_proj \
    --lora_bias none \
    --task_type CAUSAL_LM \
    --load_in_4bit \
    --bnb_4bit_quant_type nf4 \
    --bnb_4bit_use_double_quant \
    --bnb_4bit_compute_dtype "$DTYPE" \
    --bnb_4bit_quant_storage "$DTYPE" \
    --optim_type bnb_adamw8bit \
    --loraplus_lr_ratio 16 \
    --bnb_optim_percentile_clipping 5 \
    --learning_rate 3e-5 \
    --betas 0.9 0.95 \
    --decay_method cosine \
    --warmup_steps 100 \
    --min_lr 3e-6 \
    --decay_steps "$TRAIN_STEPS" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --gradient_accum_steps "$GRADIENT_ACCUM_STEPS" \
    --model_torch_dtype "$DTYPE" \
    --mixed_precision "$DTYPE" \
    --train_steps "$TRAIN_STEPS" \
    --valid_interval 500 \
    --valid_steps 50 \
    --save_interval 500 \
    --saved_checkpoint_limit 10 \
    --wandb_logging \
    --wandb_project medical_llama2 \
    --wandb_name expr-1 \
    --wandb_logging_interval 50 \
    --push_to_hub \
    --push_tokenizer \
    --repo_id medical_llama2
