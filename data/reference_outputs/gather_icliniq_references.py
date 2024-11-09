#!/usr/bin/env python3

import jsonlines

import datasets


DEFAULT_INSTRUCTION = 'If you are a doctor, please answer the medical questions based on the patient\'s description.'

def main():
    reference_files = {
        'gpt-4': './gpt-4/gpt-4_iCliniq_output.jsonl',
        'gpt-3.5-turbo': './gpt-3.5-turbo/gpt-3.5-turbo_iCliniq_output.jsonl',
        'text-davinci-003': './text-davinci-003/text-davinci-003_iCliniq_output.jsonl',
        'claude-2': './claude-2/claude-2_iCliniq_output.jsonl',
    }

    chatdoctor_icliniq_dataset = datasets.load_dataset('lavita/ChatDoctor-iCliniq', split='train')
    chatdoctor_icliniq_dataset = chatdoctor_icliniq_dataset.select(range(1000))

    output_file = './iCliniq-1k.jsonl'
    json_data: list[dict[str, str]] = []
    with jsonlines.open(reference_files['gpt-4'], 'r') as f:
        for line in f:
            input = line['input']
            json_data.append({
                'instruction': DEFAULT_INSTRUCTION,
                'input': input,
            })

    for model_name, reference_file in reference_files.items():
        with jsonlines.open(reference_file, 'r') as f:
            for i, line in enumerate(f):
                if model_name != 'claude-2':
                    input = line['input']
                    input = input.strip()
                    if input != json_data[i]['input']:
                        raise ValueError('Found unmatched input at index {i} for model {model_name}')
                json_data[i][f'{model_name}-answer'] = line['output']if model_name != 'claude-2' else line['response']
    for idx, item in enumerate(chatdoctor_icliniq_dataset):
        json_data[idx]['icliniq-answer'] = item['answer_icliniq']

    with jsonlines.open(output_file, 'w') as f:
        f.write_all(json_data)

    print(f'Wrote {len(json_data)} entries to {output_file}')


if __name__ == '__main__':
    exit(main())
