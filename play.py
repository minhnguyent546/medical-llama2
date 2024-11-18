import argparse
import random

import torch

from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    TextStreamer,
)


class SpecialToken:
    SOS = '<s>'
    EOS = '</s>'
    UNK = '<unk>'
    PAD = EOS  # as llama does not use padding token
    START_INST = '[INST]'
    END_INST = '[/INST]'
    START_SYS = '<<SYS>>'
    END_SYS = '<</SYS>>'


LLAMA_SYSTEM_PROMPT = '''You are a helpful, respectful, and honest AI Medical Assistant trained on a vast dataset of health information. Please be thorough and provide an informative answer.

If you don\'t know the answer to a specific medical inquiry, advise seeking professional help from doctors instead of sharing false information.'''

ALPACA_SYSTEM_PROMPT = 'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request'
ALPACA_SYSTEM_PROMPT_NO_INPUT = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.'


def play(args: argparse.Namespace) -> None:
    set_seed(args.seed)

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
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    model = LlamaForCausalLM.from_pretrained(
        args.model,
        device_map=device,
        quantization_config=bnb_config,
        revision=args.model_revision,
    )

    while True:
        instruction = input('>> ')
        if args.prompt_template == 'llama2':
            prompt = generate_llama2_prompt(user_message=instruction)
        else:
            prompt = generate_alpaca_prompt(
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
            model_response = tokenizer.decode(
                output[0, len(model_inputs['input_ids'][0]):],
                skip_special_tokens=True,
            )
            model_response = model_response.strip()
            print(model_response)

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def generate_llama2_prompt(
    user_message: str,
    response: str | None = None,
) -> str:
    prompt = (
        f'{SpecialToken.SOS}{SpecialToken.START_INST} {SpecialToken.START_SYS}\n'
        f'{LLAMA_SYSTEM_PROMPT}\n'
        f'{SpecialToken.END_SYS}\n\n'
        f'{user_message} {SpecialToken.END_INST}'
    )
    if response is not None:
        prompt += f' {response}'
    return prompt.strip()

def generate_alpaca_prompt(
    instruction: str,
    input: str | None = None,
    response: str | None = None,
) -> str:
    prompt = ''
    if input is not None and input and input.strip() != '<noinput>':
        prompt = (
            f'{ALPACA_SYSTEM_PROMPT}\n\n'
            f'### Instruction:\n'
            f'{instruction}\n\n'
            f'### Input:\n'
            f'{input}\n\n'
            f'### Response: '
            f'{response}'
        )
    else:
        prompt = (
            f'{ALPACA_SYSTEM_PROMPT_NO_INPUT}\n\n'
            f'### Instruction:\n'
            f'{instruction}\n\n'
            f'### Response: '
            f'{response}'
        )
    return prompt.strip()

def add_opts(parser: argparse.ArgumentParser) -> None:
    _add_general_opts(parser)
    _add_generation_opts(parser)
    _add_bitsandbytes_opts(parser)

def _add_general_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group('General')
    group.add_argument(
        '--model',
        type=str,
        help='Name or path to the model to use',
        default='minhnguyent546/Med-Alpaca-2-7b-chat',
    )
    group.add_argument(
        '--model_revision',
        type=str,
        help='Model checkpoint revision',
        default='main',
    )
    group.add_argument(
        '--seed',
        type=int,
        help='Seed for random number generators',
        default=1061109567,
    )
    group.add_argument(
        '--prompt_template',
        type=str,
        help='Which prompt template to use',
        choices=['llama2', 'alpaca'],
        default='llama2',
    )
    group.add_argument(
        '--device',
        type=str,
        help='which device to use for generating',
        default='auto',
    )

def _add_generation_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group('Generation')
    group.add_argument(
        '--streaming',
        action='store_true',
        help='Whether to stream the model response',
    )
    group.add_argument(
        '--do_sample',
        action='store_true',
        help='Whether to do sampling when generating text',
    )
    group.add_argument(
        '--max_new_tokens',
        type=int,
        help='Maximum number of new tokens',
    )
    group.add_argument(
        '--temperature',
        type=float,
        help='Temperature for generation',
    )
    group.add_argument(
        '--top_k',
        type=int,
        help='Number of highest probability vocabulary tokens to keep (top-k filtering, 0 means deactivate top_k sampling)',
    )
    group.add_argument(
        '--top_p',
        type=float,
        help='Keep the top tokens with cumulative probability >= top_p (nucleus filtering)',
    )
    group.add_argument(
        '--num_beams',
        type=int,
        help='Number of beams for beam search',
        default=1,
    )
    group.add_argument(
        '--generation_early_stopping',
        action='store_true',
        help='If set to `True` beam search is stopped when at least `num_beams` sentences finished per batch',
    )
    group.add_argument(
        '--no_repeat_ngram_size',
        type=int,
        help='If set to int > 0, all ngrams of size `no_repeat_ngram_size` can only occur once',
    )
    group.add_argument(
        '--num_return_sequences',
        type=int,
        help='Number of highest scoring sequences to return',
        default=1,
    )
    group.add_argument(
        '--repetition_penalty',
        type=float,
        help='Penalty for repetition',
    )

def _add_bitsandbytes_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group('bitsandbytes config')
    group.add_argument(
        '--load_in_8bit',
        action='store_true',
        help='Enable 8-bit quantization in bitsandbytes with LLM.int8()',
    )
    group.add_argument(
        '--load_in_4bit',
        action='store_true',
        help='Enable 4-bit quantization in bitsandbytes by replacing the Linear layers with FP4/NF4 layers from `bitsandbytes`',
    )
    group.add_argument(
        '--llm_int8_threshold',
        type=float,
        help='Outlier threshold for outlier detection as described in LLM.int8()',
        default=6.0,
    )
    group.add_argument(
        '--llm_int8_skip_modules',
        nargs='*',
        type=str,
        help='An explicit list of the modules that we do not want to convert in 8-bit (e.g. `lm_head` in CausalLM)',
    )
    group.add_argument(
        '--llm_int8_enable_fp32_cpu_offload',
        action='store_true',
        help='If you want to split your model in different parts and run some parts in int8 on GPU and some parts in fp32 on CPU',
    )
    group.add_argument(
        '--llm_int8_has_fp16_weight',
        action='store_true',
        help='This flag runs LLM.int8() with 16-bit main weights. This is useful for fine-tuning as the weights do not have to be converted back and forth for the backward pass',
    )
    group.add_argument(
        '--bnb_4bit_quant_type',
        type=str,
        help='bitandbytes 4-bit quantization type',
        choices=["fp4", "nf4"],
        default="fp4",
    )
    group.add_argument(
        '--bnb_4bit_use_double_quant',
        action='store_true',
        help='Whether to use nested quantization (i.e. the quantization constants from the first quantization are quantized again)',
    )
    group.add_argument(
        '--bnb_4bit_compute_dtype',
        type=str,
        help='bitsandbytes 4-bit compute dtype',
        default="float16",
    )
    group.add_argument(
        '--bnb_4bit_quant_storage',
        type=str,
        help='bitsandbytes 4-bit quantization storage dtype (should be the same as --model_torch_dtype)',
        default="float16",
    )

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Playing with Med-Alpaca-2-7b-chat',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_opts(parser)
    args = parser.parse_args()
    play(args)


if __name__ == '__main__':
    main()
