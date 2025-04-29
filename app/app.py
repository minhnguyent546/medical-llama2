import gc
import os

import gradio as gr

from llama_cpp import Llama


ALPACA_SYSTEM_PROMPT = 'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request'
ALPACA_SYSTEM_PROMPT_NO_INPUT = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.'

DEFAULT_MODEL = 'Med-Alpaca-2-7b-chat.Q4_K_M'

model_paths = {
    'Med-Alpaca-2-7b-chat.Q2_K': {
        'repo_id': 'minhnguyent546/Med-Alpaca-2-7b-chat-GGUF',
        'filename': 'Med-Alpaca-2-7B-chat.Q2_K.gguf',
    },
    'Med-Alpaca-2-7b-chat.Q4_K_M': {
        'repo_id': 'minhnguyent546/Med-Alpaca-2-7b-chat-GGUF',
        'filename': 'Med-Alpaca-2-7B-chat.Q4_K_M.gguf',
    },
    'Med-Alpaca-2-7b-chat.Q6_K': {
        'repo_id': 'minhnguyent546/Med-Alpaca-2-7b-chat-GGUF',
        'filename': 'Med-Alpaca-2-7B-chat.Q6_K.gguf',
    },
    'Med-Alpaca-2-7b-chat.Q8_0': {
        'repo_id': 'minhnguyent546/Med-Alpaca-2-7b-chat-GGUF',
        'filename': 'Med-Alpaca-2-7B-chat.Q8_0.gguf',
    },
    'Med-Alpaca-2-7b-chat.F16': {
        'repo_id': 'minhnguyent546/Med-Alpaca-2-7b-chat-GGUF',
        'filename': 'Med-Alpaca-2-7B-chat.F16.gguf',
    },
}
model = None

def generate_alpaca_prompt(
    instruction: str,
    input: str | None = None,
    response: str = '',
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

def chat_completion(
    message,
    history,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    repeatition_penalty: float,
    top_k: int,
    top_p: float,
):
    global model
    if model is None:
        reload_model(DEFAULT_MODEL)
    prompt = generate_alpaca_prompt(instruction=message)
    response_iterator = model(
        prompt,
        stream=True,
        seed=seed,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=repeatition_penalty,
    )
    partial_response = ''
    for token in response_iterator:
        partial_response += token['choices'][0]['text']
        yield partial_response

def reload_model(model_name: str):
    global model
    model = None
    gc.collect()
    model = Llama.from_pretrained(
        **model_paths[model_name],
        n_ctx=4096,
        n_threads=4,
        cache_dir='./.cache/huggingface'
    )

    app_title_mark = gr.Markdown(f"""<center><font size=18>{model_name}</center>""")
    chatbot = gr.Chatbot(
        type='messages',
        height=500,
        placeholder='<strong>Hi doctor, I have a headache, what should I do?</strong>',
        label=model_name,
        avatar_images=[None, 'https://raw.githubusercontent.com/minhnguyent546/medical-llama2/refs/heads/master/assets/medical_llama2.png'],  # pyright: ignore[reportArgumentType]
    )
    return app_title_mark, chatbot

def main() -> None:
    with gr.Blocks(theme=gr.themes.Ocean()) as demo:
        app_title_mark = gr.Markdown(f"""<center><font size=18>{DEFAULT_MODEL}</center>""")

        model_options = list(model_paths.keys())

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    model_radio = gr.Radio(choices=model_options, label='Model', value=DEFAULT_MODEL)
                with gr.Row():
                    seed = gr.Number(value=998244353, label='Seed')
                    max_new_tokens = gr.Number(value=512, minimum=64, maximum=2048, label='Max new tokens')

                with gr.Row():
                    temperature = gr.Slider(0, 2, step=0.01, label='Temperature', value=0.6)
                    repeatition_penalty = gr.Slider(0.01, 5, step=0.05, label='Repetition penalty', value=1.1)

                with gr.Row():
                    top_k = gr.Slider(1, 100, step=1, label='Top k', value=40)
                    top_p = gr.Slider(0, 1, step=0.01, label='Top p', value=0.9)

            with gr.Column(scale=5):
                chatbot = gr.Chatbot(
                    type='messages',
                    height=500,
                    placeholder='<strong>Hi doctor, I have a headache, what should I do?</strong>',
                    label=DEFAULT_MODEL,
                    avatar_images=[None, 'https://raw.githubusercontent.com/minhnguyent546/medical-llama2/refs/heads/master/assets/medical_llama2.png'],  # pyright: ignore[reportArgumentType]
                )
                textbox = gr.Textbox(
                    placeholder='Hi doctor, I have a headache, what should I do?',
                    container=False,
                    submit_btn=True,
                    stop_btn=True,
                )

                chat_interface = gr.ChatInterface(
                    chat_completion,
                    type='messages',
                    chatbot=chatbot,
                    textbox=textbox,
                    additional_inputs=[
                        seed,
                        max_new_tokens,
                        temperature,
                        repeatition_penalty,
                        top_k,
                        top_p,
                    ],
                )

        model_radio.change(reload_model, inputs=[model_radio], outputs=[app_title_mark, chatbot])

        demo.queue(api_open=False, default_concurrency_limit=20)
        demo.launch(max_threads=5, share=os.environ.get('GRADIO_SHARE', False))


if __name__ == '__main__':
    main()
