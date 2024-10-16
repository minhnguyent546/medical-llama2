from typing import Any

import torch
from torch.utils.data import Dataset

import datasets
from transformers import LlamaTokenizer

from medical_llama2.constants import SYSTEM_PROMPT


class MedicalDataset(Dataset):
    def __init__(
        self,
        dataset: datasets.Dataset,
        tokenizer: LlamaTokenizer,
        seq_length: int,
        train_on_inputs: bool = True,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.train_on_inputs = train_on_inputs

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | list[Any]]:
        question = self.dataset[index]['Patient']
        answer = self.dataset[index]['Doctor']
        prompt = self.tokenizer.apply_chat_template([
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': answer},
        ], tokenize=False)
        tokenized_prompt = self.tokenizer(
            prompt,
            max_length=self.seq_length,
            truncation=True,
            padding=False,
        )
        input_ids = tokenized_prompt['input_ids'][:-1]
        attention_mask = tokenized_prompt['attention_mask'][:-1]
        labels = tokenized_prompt['input_ids'][1:]

        if not self.train_on_inputs:
            user_prompt = self.tokenizer.apply_chat_template([
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': question},
            ], tokenize=False)
            tokenized_user_prompt = self.tokenizer(
                user_prompt,
                max_length=self.seq_length,
                truncation=True,
                padding=False,
            )
            user_prompt_token_len = len(tokenized_user_prompt['input_ids'])
            user_prompt_token_len -= 1  # as tokens in labels are shifted one token to the left

            labels = [self.tokenizer.pad_token_id] * user_prompt_token_len \
                    + labels[user_prompt_token_len:]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.int64),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.int64),
        }
