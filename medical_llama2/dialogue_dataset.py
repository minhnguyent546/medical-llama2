from typing import Any, Literal

import torch
from torch.utils.data import Dataset

import datasets
from transformers import LlamaTokenizer

import medical_llama2.utils as utils
from medical_llama2.constants import (
    ALPACA_SYSTEM_PROMPT,
    LLAMA_SYSTEM_PROMPT,
)


class DialogueDataset(Dataset):
    def __init__(
        self,
        dataset: datasets.Dataset,
        tokenizer: LlamaTokenizer,
        seq_length: int,
        input_field: str,
        output_field: str,
        instruction_field: str | None = None,
        train_on_inputs: bool = True,
        prompt_template: Literal['llama2', 'alpaca'] = 'llama2',
        dataset_num_procs: int = 4,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.input_field = input_field
        self.output_field = output_field
        self.instruction_field = instruction_field
        self.train_on_inputs = train_on_inputs
        self.prompt_template = prompt_template
        self.dataset_num_procs = dataset_num_procs

        if self.prompt_template == 'alpaca':
            assert self.instruction_field is not None, \
                '`instruction_field` must be provided when using the alpaca prompt template'
        self.preproces()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | list[Any]]:
        input_ids = self.dataset[index]['input_ids']
        labels = self.dataset[index]['labels']
        attention_mask = self.dataset[index]['attention_mask']
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }

    def preproces(self) -> None:
        self.dataset = self.dataset.map(
            self._tokenize,
            desc='Preparing and tokenizing prompts',
            num_proc=self.dataset_num_procs,
        )

    def _tokenize(self, item: dict[str, Any]) -> dict[str, Any]:
        if self.prompt_template == 'llama2':
            prompt = utils.generate_llama2_prompt(
                user_message=item[self.input_field],
                response=item[self.output_field],
            )
        else:
            prompt = utils.generate_alpaca_prompt(
                instruction=item[self.instruction_field],
                input=item[self.input_field],
                response=item[self.output_field],
            )
        tokenized_prompt = self.tokenizer(
            prompt,
            max_length=self.seq_length,
            truncation=True,
            padding=False,
        )
        if (
            tokenized_prompt['input_ids'][-1] != self.tokenizer.eos_token_id and
            len(tokenized_prompt['input_ids']) < self.seq_length
        ):
            tokenized_prompt['input_ids'].append(self.tokenizer.eos_token_id)
            tokenized_prompt['attention_mask'].append(1)

        input_ids = tokenized_prompt['input_ids']
        attention_mask = tokenized_prompt['attention_mask']
        labels = tokenized_prompt['input_ids'].copy()
        if not self.train_on_inputs:
            # if we not train on input tokens, mask them out
            if self.prompt_template == 'llama2':
                user_prompt = utils.generate_llama2_prompt(
                    user_message=item[self.input_field],
                )
            else:
                user_prompt = utils.generate_alpaca_prompt(
                    instruction=item[self.instruction_field],
                    input=item[self.input_field],
                    response='',
                )
            tokenized_user_prompt = self.tokenizer(
                user_prompt,
                max_length=self.seq_length,
                truncation=True,
                padding=False,
            )
            user_prompt_token_len = len(tokenized_user_prompt['input_ids'])
            labels = [self.tokenizer.pad_token_id] * user_prompt_token_len \
                + labels[user_prompt_token_len:]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
