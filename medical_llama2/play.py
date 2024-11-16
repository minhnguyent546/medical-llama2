import argparse

import torch

from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    TextStreamer,
)

import medical_llama2.opts as opts
import medical_llama2.utils as utils


def play(args: argparse.Namespace) -> None:
    utils.set_seed(args.seed)

    tokenizer: LlamaTokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    is_load_in_kbit = args.load_in_8bit or args.load_in_4bit
    bnb_config = None
    if device.type == 'cuda' and is_load_in_kbit:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
            llm_int8_threshold=args.llm_int8_threshold,
            llm_int8_skip_modules=args.llm_int8_skip_modules,
            llm_int8_enable_fp32_cpu_offload=args.llm_int8_enable_fp32_cpu_offload,
            llm_int8_has_fp16_weight=args.llm_int8_has_fp16_weight,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
            bnb_4bit_quant_storage=args.bnb_4bit_quant_storage,
        )

    streamer = None
    if args.streaming:
        streamer = TextStreamer(tokenizer, skip_prompt=True)
    model = LlamaForCausalLM.from_pretrained(
        args.model,
        device_map=device,
        quantization_config=bnb_config,
        revision=args.model_revision,
    )

    while True:
        instruction = input('>> ')
        if args.prompt_template == 'llama2':
            prompt = utils.generate_llama2_prompt(user_message=instruction)
        else:
            prompt = utils.generate_alpaca_prompt(
                instruction=instruction,
                response='',
            )
        model_inputs = tokenizer([prompt], return_tensors='pt').to(device)
        output = model.generate(
            **model_inputs,
            do_sample=args.do_sample,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_beams=args.num_beams,
            early_stopping=args.generation_early_stopping,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            num_return_sequences=args.num_return_sequences,
            repetition_penalty=args.repetition_penalty,
            streamer=streamer,
        )
        if not args.streaming:
            model_response = tokenizer.decode(output[0, len(model_inputs['input_ids'][0]):], skip_special_tokens=True)
            model_response = model_response.strip()
            print(model_response)

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Playing with LLama-2',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_opts(parser)
    args = parser.parse_args()
    play(args)


def add_opts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--model',
        type=str,
        help='Name or path to the model to use',
        default='minhnguyent546/Med-Alpaca-2-7b-chat',
    )
    parser.add_argument(
        '--model_revision',
        type=str,
        help='Model checkpoint revision',
        default='main',
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Seed for random number generators',
        default=1061109567,
    )
    parser.add_argument(
        '--prompt_template',
        type=str,
        help='Which prompt template to use',
        choices=['llama2', 'alpaca'],
        default='llama2',
    )
    parser.add_argument(
        '--device',
        type=str,
        help='which device to use for generating',
        default='auto',
    )
    parser.add_argument(
        '--streaming',
        action='store_true',
        help='Whether to stream the model response',
    )
    opts._add_generation_opts(parser)  # pyright: ignore[reportPrivateUsage]
    opts._add_bitsandbytes_opts(parser)  # pyright: ignore[reportPrivateUsage]


if __name__ == '__main__':
    main()
