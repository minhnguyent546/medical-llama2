import textwrap
from typing import Any

import torch
from torch.utils.data import Dataset

from transformers import LlamaTokenizer

from medical_llama2.constants import SYSTEM_PROMPT, SpecialToken


class MedicalDataset(Dataset):
    def __init__(self, dataset, tokenizer: LlamaTokenizer, seq_length: int):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_length = seq_length

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | list[Any]]:
        question = self.dataset[index]['Patient']
        answer = self.dataset[index]['Doctor']
        prompt = generate_prompt(question=question, answer=answer, system_prompt=SYSTEM_PROMPT)

        tokenized_prompt = self.tokenizer(
            prompt,
            max_length=self.seq_length,
            truncation=True,
            padding=False,
        )

        input_ids = tokenized_prompt['input_ids'][:-1]
        attention_mask = tokenized_prompt['attention_mask'][:-1]
        labels = tokenized_prompt['input_ids'][1:]
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.int64),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.int64),
        }

def generate_prompt(question: str, answer: str, system_prompt: str | None = None) -> str:
    prompt = textwrap.dedent(f'''
        {SpecialToken.SOS}{SpecialToken.START_INST} {SpecialToken.START_SYS}
        {system_prompt}
        {SpecialToken.END_SYS}

        Question: {question}
        Answer: {answer} {SpecialToken.END_INST}{SpecialToken.EOS}''')

    return prompt.strip()
