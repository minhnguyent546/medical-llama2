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
| Med-Alpaca-2-7b-chat | [HuggingFace](https://huggingface.co/minhnguyent546/Med-Alpaca-2-7b-chat/settings) |
| Med-Alpaca-2-7b-chat-LoRA | [HuggingFace](https://huggingface.co/minhnguyent546/Med-Alpaca-2-7b-chat-LoRA) |
| Med-Alpaca-2-7b-chat GGUFs | [Huggingface](https://huggingface.co/minhnguyent546/Med-Alpaca-2-7b-chat-GGUF) |
| Alpaca-Llama-2-7b-chat | [HuggingFace](https://huggingface.co/minhnguyent546/Alpaca-Llama-2-7b-chat) |
| Alpaca-Llama-2-7b-chat-LoRA | [HuggingFace](https://huggingface.co/minhnguyent546/Alpaca-Llama-2-7b-chat-LoRA) |
| Alpaca-Llama-2-7b-chat GGUFs | [Huggingface](https://huggingface.co/minhnguyent546/Alpaca-Llama-2-7b-chat-GGUF) |

**Prompt template**: note that all of the models above use **Alpaca** prompt template. Refer to [this](https://github.com/tatsu-lab/stanford_alpaca) for more information.

## Training

**TODO**

*Updating...*
