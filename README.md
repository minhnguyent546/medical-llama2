<p align="center"><image src="./assets/medical_llama2.png" alt="medical-llama2" width="500px"></p>

# Medical LLaMA 2

A medical chatbot utilizing Llama-2 7B chat model.

## Evaluation results

Models are evaluated with GPT-4 score using GPT-4o mini.

- GPT-4 scores for **Alpaca-Llama-2-7b-chat** on MedInstruct-test:

| Ref. response | Win rate | Win rate (non-tied) | Win rate (w. half scores) | Consistency | Error |
| :---: | :---: | :---: | :---: | :---: | :---: |
| GPT-4 | 25.9 | 32.7 | 36.3 | 79.2% | 0.0 | 
| GPT-3.5-turbo | 31.9 | 37.9 | 39.8 | 84.3% | 0.0 |
| Text-davinci-003 | 86.1 | 92.1 | 89.4 | 93.5% | 0.0 |
| Claude-2 | 4.2 | 5.9 | 19.0 | 70.4% | 0.0 | 

- GPT-4 scores for **Med-Alpaca-2-7b-chat** on MedInstruct-test:

| Ref. response | Win rate | Win rate (non-tied) | Win rate (w. half scores) | Consistency | Error |
| :---: | :---: | :---: | :---: | :---: | :---: |
| GPT-4 | 29.6 | 41.8 | 44.2 | 70.8% | 0.0 |
| GPT-3.5-turbo | 31.9 | 46.9 | 47.9 | 68.1% | 0.0 |
| Text-davinci-003 | 87.5 | 94.5 | 91.2 | 92.6% | 0.0 |
| Claude-2 | 8.8 | 11.5 | 20.6 | 76.4% | 0.0 |

## Pre-trained models

All of the pre-trained model weights can be found in the table below.

| Model | link |
| --- | :---: |
| Med-Alpaca-2-7b-chat | [Huggingface](https://huggingface.co/minhnguyent546/Med-Alpaca-2-7b-chat) |
| Med-Alpaca-2-7b-chat-LoRA | [Huggingface](https://huggingface.co/minhnguyent546/Med-Alpaca-2-7b-chat-LoRA) |
| Med-Alpaca-2-7b-chat GGUFs | [Huggingface](https://huggingface.co/minhnguyent546/Med-Alpaca-2-7b-chat-GGUF) |
| Alpaca-Llama-2-7b-chat | [Huggingface](https://huggingface.co/minhnguyent546/Alpaca-Llama-2-7b-chat) |
| Alpaca-Llama-2-7b-chat-LoRA | [Huggingface](https://huggingface.co/minhnguyent546/Alpaca-Llama-2-7b-chat-LoRA) |
| Alpaca-Llama-2-7b-chat GGUFs | [Huggingface](https://huggingface.co/minhnguyent546/Alpaca-Llama-2-7b-chat-GGUF) |

**Prompt template**: note that all of the models above use **Alpaca** prompt template. Refer to [this](https://github.com/tatsu-lab/stanford_alpaca) for more information.

## Setup

- Clone this repo:
```bash
git clone https://github.com/minhnguyent546/medical-llama2.git
cd medical-llama2
```

- Create a virtual environment with conda:
```bash
conda create -n medical-llama2 python=3.10 -y 
conda activate medical-llama2
pip install -r requirements.txt
```

- Or if you prefer Docker:
```bash
docker build -t medical-llama2 .
docker run \
    --rm \
    --shm-size=512mb \
    --gpus=all \
    -e HF_TOKEN='<YOUR_HF_TOKEN>' \
    -e WANDB_API_KEY='<YOUR_WANDB_API_KEY>' \
    -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    medical-llama2 run_finetune.py --help
```

## Training

The model was trained using 4 RTX 4090 GPUs with 24GB of VRAM each. The training process took about 1 hour to finish.

You can run the following command to train the model:

```bash
python run_finetune.py \
    --do_train \
    --checkpoints_dir ./checkpoints \
    --seed 998244353 \
    --dataset_path lavita/AlpaCare-MedInstruct-52k \
    --input_field input \
    --output_field output \
    --instruction_field instruction \
    --prompt_template alpaca \
    --validation_size 500 \
    --test_size 1000 \
    --dataset_num_procs 4 \
    --model_checkpoint minhnguyent546/Alpaca-Llama-2-7b-chat \
    --tokenizer_checkpoint minhnguyent546/Alpaca-Llama-2-7b-chat \
    --model_checkpoint_revision 56fbdb8c156a197b0c3f9a80442b667f454cbbb6 \
    --max_seq_length 512 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --lora_r 16 \
    --lora_target_modules q_proj v_proj k_proj o_proj gate_proj down_proj up_proj \
    --lora_bias none \
    --task_type CAUSAL_LM \
    --load_in_4bit \
    --bnb_4bit_compute_dtype float16 \
    --bnb_4bit_quant_storage float16 \
    --optim_type bnb_adamw32bit \
    --loraplus_lr_ratio 16 \
    --bnb_optim_percentile_clipping 5 \
    --learning_rate 1e-5 \
    --min_lr 1e-6 \
    --betas 0.9 0.95 \
    --decay_method cosine \
    --warmup_steps 36 \
    --decay_steps 800 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --gradient_accum_steps 16 \
    --model_torch_dtype float16 \
    --mixed_precision float16 \
    --max_grad_norm 1.0 \
    --train_steps 800 \
    --valid_interval 100 \
    --valid_steps 25 \
    --save_interval 100 \
    --saved_checkpoint_limit 30 \
    --ddp_timeout 2400 \
    --wandb_logging \
    --wandb_project medical-llama2 \
    --wandb_name test \
    --wandb_logging_interval 10 \
    --push_to_hub \
    --push_at_steps 400,800 \
    --push_tokenizer \
    --repo_id medical-llama2 \
    --commit_message test
```

## Demo

You can try out the demo in HuggingFace Spaces [here](https://huggingface.co/spaces/minhnguyent546/Med-Alpaca-2-7b-chat). To run the model locally, please prefer to [app](./app) for more information.
