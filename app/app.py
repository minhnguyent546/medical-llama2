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
    assert model is not None
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

def load_css(file_path: str) -> str:
    css_content = ''
    try:
        with open(file_path, 'r') as file:
            css_content = file.read()
    except OSError:
        print(f'Failed to load CSS file: {file_path}')
    return css_content

with gr.Blocks(theme=gr.themes.Ocean(), css=load_css('./css/styles.css')) as demo:
    app_title_mark = gr.Markdown(f"""<center><font size=18>{DEFAULT_MODEL}</center>""")

    model_options = list(model_paths.keys())

    with gr.Row():
        with gr.Column(scale=2, min_width=0) as config_column:
            config_visible = gr.State(True)

            with gr.Row():
                toggle_config_btn = gr.Button('☰', variant='secondary', size='sm', min_width=40)

            with gr.Column(visible=True) as config_panel:
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

        with gr.Column(scale=5) as chat_column:
            chatbot = gr.Chatbot(
                type='messages',
                height=500,
                placeholder='<strong>Hi doctor, I have a headache, what should I do?</strong>',
                label=DEFAULT_MODEL,
                avatar_images=[None, 'https://raw.githubusercontent.com/minhnguyent546/medical-llama2/refs/heads/master/assets/medical_llama2.png'],  # pyright: ignore[reportArgumentType]
                elem_classes='max-width-chat',
            )
            textbox = gr.Textbox(
                placeholder='Hi doctor, I have a headache, what should I do?',
                container=False,
                submit_btn=True,
                stop_btn=True,
                elem_classes='max-width-chat',
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

    def toggle_config(visible_state):
        new_visible = not visible_state
        if new_visible:
            # Config panel visible: normal layout, no max-width constraint
            config_scale = 2
            chat_scale = 5
            chat_css_class = ''
        else:
            # Config panel collapsed: chat takes more space, apply max-width constraint
            config_scale = 0
            chat_scale = 7
            chat_css_class = 'max-width-chat'

        print(f'{chat_css_class = }')
        return (
            new_visible,
            gr.update(visible=new_visible),
            gr.update(scale=config_scale),
            gr.update(scale=chat_scale),
            gr.update(elem_classes=chat_css_class),
            gr.update(elem_classes=chat_css_class),
        )

    toggle_config_btn.click(
        toggle_config,
        inputs=[config_visible],
        outputs=[
            config_visible,
            config_panel,
            config_column,
            chat_column,
            chatbot,
            textbox,
        ]
    )

    model_radio.change(reload_model, inputs=[model_radio], outputs=[app_title_mark, chatbot])

    demo.queue(api_open=False, default_concurrency_limit=20)
    demo.launch(max_threads=5, share=os.environ.get('GRADIO_SHARE', False))
