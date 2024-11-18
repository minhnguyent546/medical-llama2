#!/usr/bin/env bash

python play.py \
  --model minhnguyent546/Med-Alpaca-2-7b-chat \
  --seed 42 \
  --prompt_template alpaca \
  --streaming \
  --do_sample \
  --max_new_tokens 256 \
  --temperature 0.6 \
  --top_k 40 \
  --top_p 0.9 \
  --repetition_penalty 1.1 \
  --load_in_4bit \
  --bnb_4bit_compute_dtype float16 \
  --bnb_4bit_quant_storage float16
