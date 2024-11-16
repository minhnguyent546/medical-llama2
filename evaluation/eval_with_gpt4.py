import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import jsonlines

from tqdm.autonotebook import tqdm

import openai
from openai.types.chat.chat_completion import ChatCompletion


API_KEY = os.getenv('OPENAI_API_KEY')
SYSTEM_PROMPT = 'You are a helpful and precise instruction-following assistant for checking the quality of the responses for a given user instruction.'

client = openai.OpenAI(api_key=API_KEY)

class Verdict(Enum):
    WIN = 'WIN'
    LOSE = 'LOSE'
    TIE = 'TIE'
    UNDETERMINED = 'UNDETERMINED'

@dataclass
class EvalResult:
    model_output_id: int
    prompt: str = ''
    final_verdict: Verdict = Verdict.UNDETERMINED
    extracted_verdict: str = ''
    response: str = ''
    model_response_first: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            'model_output_id': self.model_output_id,
            'model_response_first': self.model_response_first,
            'prompt': self.prompt,
            'response': self.response,
            'final_verdict': self.final_verdict.value,
            'extracted_verdict': self.extracted_verdict,
        }

def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    model_output_records = []
    with jsonlines.open(args.model_outputs_file, 'r') as f:
        for line in f:
            model_output_records.append(line)

    with open('./pairwise_comparison_template.txt', 'r', encoding='utf-8') as f:
        prompt_template = f.read().strip()

    eval_results: list[EvalResult] = []
    progress_bar = tqdm(model_output_records, desc='Evaluating', unit='record')
    for record in progress_bar:
        prompt = make_prompt(
            prompt_template=prompt_template,
            instruction=record['instruction'],
            answer_a=record['response'] if args.model_response_first else record['reference'],
            answer_b=record['response'] if not args.model_response_first else record['reference'],
            input=record['input'],
        )
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt},
        ]

        response = make_request_with_retry(messages, args.num_api_call_retries, args)
        if response is not None:
            eval_results.append(get_eval_result(response, prompt, record['id'], args.model_response_first))
        else:
            eval_results.append(EvalResult(model_output_id=record['id']))

    eval_results_file_path, ext = os.path.splitext(args.eval_results_file)
    if not ext:
        ext = '.json'
    cur_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
    eval_results_file_path = f'{eval_results_file_path}_{args.model}_{cur_datetime}'
    if args.model_response_first:
        eval_results_file_path += '_model_first'
    else:
        eval_results_file_path += '_reference_first'
    eval_results_file_path += ext
    with open(eval_results_file_path, 'w') as f:
        eval_results_serializable = [eval_result.as_dict() for eval_result in eval_results]
        json.dump(eval_results_serializable, f, indent=4)

def make_prompt(
    prompt_template: str,
    instruction: str,
    answer_a: str,
    answer_b: str,
    input: str | None = None,
) -> str:
    if input is not None and input != '' and input != '<noinput>':
        instruction = f'{instruction}\nInput: {input}'
    return prompt_template.format(
        user_instruction=instruction,
        answer_a=answer_a,
        answer_b=answer_b,
        input=input or '',
    )

def make_request_with_retry(messages, num_retries: int, args: argparse.Namespace) -> ChatCompletion | None:
    wait_time = 10
    for _ in range(num_retries):
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_completion_tokens=args.max_completion_tokens,
                presence_penalty=0,
                frequency_penalty=0,
                seed=args.seed,
            )
            return response
        except openai.OpenAIError as err:
            print('Error processing request: ', err)
            print(f'Retry in {wait_time} seconds')
            time.sleep(wait_time)
            wait_time = int(wait_time ** 1.15)
        except Exception as err:
            print('Unexpected error while making request: ', err)

    return None

def get_eval_result(
    response: ChatCompletion,
    prompt: str,
    model_output_id: int,
    model_response_first: bool,
) -> EvalResult:
    eval_result = EvalResult(model_output_id=model_output_id, prompt=prompt, model_response_first=args.model_response_first)
    response_content = response.choices[0].message.content
    eval_result.response = response_content

    extracted_verdict = re.findall(r'\[\[(.*?)\]\]', response_content)
    if extracted_verdict and extracted_verdict[0] in ('A', 'B', 'C'):
        extracted_verdict = extracted_verdict[0]
        eval_result.extracted_verdict = extracted_verdict
        if extracted_verdict == 'C':
            eval_result.final_verdict = Verdict.TIE
        else:
            if (extracted_verdict == 'A' and model_response_first) or (extracted_verdict == 'B' and not model_response_first):
                eval_result.final_verdict = Verdict.WIN
            else:
                eval_result.final_verdict = Verdict.LOSE

    return eval_result

def set_seed(seed: int) -> None:
    random.seed(seed)

def add_opts(parser) -> None:
    parser.add_argument(
        '--seed',
        type=int,
        help='Seed value for random number generators',
        default=998244353,
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Model to use (e.g. gpt-4o-mini-2024-07-18)',
        default='gpt-4o-mini-2024-07-18',
    )
    parser.add_argument(
        '--model_outputs_file',
        type=str,
        help='A jsonl file containing instruction, model response and reference (generated by `run_finetune.py`)',
        default='model_outputs.jsonl',
    )
    parser.add_argument(
        '--model_response_first',
        action='store_true',
        help='Whether to put the model output first (i.e. Assistant A\'s answer)',
    )
    parser.add_argument(
        '--eval_results_file',
        type=str,
        help='File path (.json file) to save the evaluation results',
        default='eval_results.json',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        help='Temperature value for sampling',
        default=1.0,
    )
    parser.add_argument(
        '--max_completion_tokens',
        type=int,
        help='Maximum number of tokens to generate by the model',
        default=512,
    )
    parser.add_argument(
        '--num_api_call_retries',
        type=int,
        help='Number of retries for API calls',
        default=20,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluating model responses with GPT-4o models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_opts(parser)
    args = parser.parse_args()
    main(args)
