import argparse


def add_run_finetune_opts(parser: argparse.ArgumentParser) -> None:
    _add_general_opts(parser)
    _add_dataset_opts(parser)
    _add_model_opts(parser)
    _add_lora_opts(parser)
    _add_bitsandbytes_opts(parser)
    _add_common_training_opts(parser)
    _add_generation_opts(parser)
    _add_wandb_opts(parser)
    _add_ddp_training_opts(parser)

def _add_general_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group('General')
    group.add_argument(
        '--checkpoints_dir',
        type=str,
        help='Directory to save model checkpoints',
        default='checkpoints',
    )
    group.add_argument(
        '--seed',
        type=int,
        help='Seed for random number generators',
        default=1061109567,
    )

def _add_dataset_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group('Dataset')
    group.add_argument(
        '--dataset_path',
        type=str,
        help='Path of the dataset to load (can be a path to a huggingface\'s dataset or a json, csv, parquet, etc files)',
    )
    group.add_argument(
        '--dataset_name',
        type=str,
        help='Name of the dataset',
    )
    group.add_argument(
        '--dataset_data_files',
        nargs='*',
        type=str,
        help='Path to the dataset files if using local files (e.g. json, csv, etc)',
    )
    group.add_argument(
        '--dataset_num_procs',
        type=int,
        help='Number of processes to use for loading the dataset',
        default=4,
    )
    group.add_argument(
        '--input_field',
        type=str,
        help='Name of the input field of the dataset (e.g. field containing user message)',
        default='input',
    )
    group.add_argument(
        '--output_field',
        type=str,
        help='Name of the output field of the dataset (e.g. field containing model response)',
        default='output',
    )
    group.add_argument(
        '--instruction_field',
        type=str,
        help='Name of the instruction field of the dataset (e.g. field containing instruction when using Alpaca dataset)',
    )
    group.add_argument(
        '--test_size',
        type=int,
        help='Test size',
        default=3_000,
    )
    group.add_argument(
        '--validation_size',
        type=int,
        help='Test size',
        default=3_000,
    )
    group.add_argument(
        '--drop_last',
        help='Whether to drop the last incomplete batch',
        action='store_true',
    )

def _add_model_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group('Model')
    group.add_argument(
        '--model_checkpoint',
        type=str,
        help='Path to the HuggingFace model checkpoint being used',
        default='meta-llama/Llama-2-7b-hf',
    )
    group.add_argument(
        '--tokenizer_checkpoint',
        type=str,
        help='Path to the HuggingFace tokenizer checkpoint being used',
        default='meta-llama/Llama-2-7b-hf',
    )
    group.add_argument(
        '--seq_length',
        type=int,
        help='Maximum sequence length',
        default=512,
    )

def _add_wandb_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group('Wandb')
    group.add_argument(
        '--wandb_logging',
        action='store_true',
        help='Enable logging to wandb',
    )
    group.add_argument(
        '--wandb_project',
        type=str,
        help='Project name',
        default='medical-llama2',
    )
    group.add_argument(
        '--wandb_name',
        type=str,
        help='Experiment name',
        default='base',
    )
    group.add_argument(
        '--wandb_logging_interval',
        type=int,
        help='Time between syncing metrics to wandb',
        default=500,
    )
    group.add_argument(
        '--wandb_resume_id',
        type=str,
        help='Id to resume a run from',
    )
    group.add_argument(
        '--wandb_notes',
        type=str,
        help='Wandb notes',
    )
    group.add_argument(
        '--wandb_tags',
        type=str,
        help='Wandb tags',
    )

def _add_common_training_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group('Training')

    # prompt template
    group.add_argument(
        '--prompt_template',
        type=str,
        help='Which prompt template to use',
        choices=['llama2', 'alpaca'],
        default='llama2',
    )

    # optimizer options
    group.add_argument(
        '--optim_type',
        type=str,
        help='Which optimizer to use',
        choices=['adam', 'adamw', 'bnb_adam32bit', 'bnb_adamw32bit', 'bnb_adam8bit', 'bnb_adamw8bit'],
        default='adamw',
    )
    group.add_argument(
        '--loraplus_lr_ratio',
        type=int,
        help='loraplus_lr_ratio to use in create_loraplus_optimizer function',
        default=16,
    )
    group.add_argument(
        '--bnb_optim_percentile_clipping',
        type=int,
        help='percentile_clipping to use in bitsandbytes\' optimizers',
        default=5,
    )
    group.add_argument(
        '--learning_rate',
        type=float,
        help='Learning rate',
        default=6.0e-4,
    )
    group.add_argument(
        '--betas',
        nargs=2,
        type=float,
        help='Optimizer betas value',
        default=[0.9, 0.999],
    )
    group.add_argument(
        '--weight_decay',
        type=float,
        help='Weight decay value',
        default=0.0,
    )

    # scheduler options
    group.add_argument(
        '--decay_method',
        type=str,
        help='Learning rate decay method (you might want to choose larger learning rate when using noam decay, e.g. 0.5)',
        choices=['cosine', 'noam'],
        default='cosine',
    )
    group.add_argument(
        '--warmup_steps',
        type=int,
        help='Warmup steps for learning rate',
        default=1_000,
    )
    group.add_argument(
        '--min_lr',
        type=float,
        help='Minimum learning rate (i.e. decay until this value) (for noam decay only)',
        default=6.0e-5,
    )
    group.add_argument(
        '--decay_steps',
        type=int,
        help='Number of steps to decay learning rate (for cosine decay only)',
        default=20_000,
    )

    # others
    group.add_argument(
        '--use_cache',
        action='store_true',
        help='Whether to enable kv-cache',
    )
    group.add_argument(
        '--train_on_inputs',
        action='store_true',
        help='Whether to train on input tokens, if set to `False` then the system and user part in the prompt will be ignored during loss calculation',
    )
    group.add_argument(
        '--train_batch_size',
        type=int,
        help='Training batch size (global batch size)',
        default=32,
    )
    group.add_argument(
        '--eval_batch_size',
        type=int,
        help='Evaluation batch size (global batch size)',
        default=32,
    )
    group.add_argument(
        '--gradient_accum_step',
        type=int,
        help='Gradient accumulation step',
        default=1,
    )
    group.add_argument(
        '--mixed_precision',
        type=str,
        help='Data type for mixed precision training',
        choices=['float16', 'bfloat16'],
    )
    group.add_argument(
        '--model_torch_dtype',
        type=str,
        help='Data type of tensors of the model',
        default='float32',
    )
    group.add_argument(
        '--train_steps',
        type=int,
        help='Number of training steps',
        default=10_000,
    )
    group.add_argument(
        '--valid_interval',
        type=int,
        help='Validation interval',
    )
    group.add_argument(
        '--valid_steps',
        type=int,
        help='Number of validation steps',
        default=50,
    )
    group.add_argument(
        '--generation_interval',
        type=int,
        help='Generation validation interval',
    )
    group.add_argument(
        '--generation_steps',
        type=int,
        help='Number of generation steps',
        default=25,
    )
    group.add_argument(
        '--bert_score_type',
        type=str,
        help='Bert score type to compute (i.e. un-scaled or scaled score)',
        choices=['unscaled', 'scaled', 'both'],
        default='scaled',
    )
    group.add_argument(
        '--save_interval',
        type=int,
        help='Steps between saving checkpoints (you should use a multiple of --valid-interval for accurate training figures (e.g. loss) when resuming from previous checkpoint)',
        default=1_000,
    )
    group.add_argument(
        '--saved_checkpoint_limit',
        type=int,
        help='Maximum number of saved checkpoints, when reached, the oldest checkpoints will be removed',
        default=10,
    )
    group.add_argument(
        '--save_model_only',
        action='store_true',
        help='Whether to save model state dict only and leave optimizer, scheduler, scaler, ....',
    )
    group.add_argument(
        '--max_grad_norm',
        type=float,
        help='Maximum gradient norm for gradient clipping (0.0 means no clipping)',
        default=0.0,
    )
    group.add_argument(
        '--use_gradient_checkpointing',
        action='store_true',
        help='Whether to use gradient checkpointing (to save memory at the expense of slower backward pass)',
    )

def _add_generation_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group('Generation')
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


def _add_ddp_training_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group('DDP training')
    group.add_argument(
        '--ddp_backend',
        type=str,
        help='DDP backend used for distributed training',
        default='nccl',
    )
def _add_lora_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group('LoRA config')
    group.add_argument(
        '--lora_alpha',
        type=int,
        help='Alpha value for LoRA scaling',
        default=8,
    )
    group.add_argument(
        '--lora_dropout',
        type=float,
        help='Dropout probability for LoRA layers',
        default=0.0,
    )
    group.add_argument(
        '--lora_r',
        type=int,
        help='LoRA matrices rank',
        default=8,
    )
    group.add_argument(
        '--lora_target_modules',
        nargs='+',
        type=str,
        help='LoRA target modules',
        default=['q_proj', 'v_proj'],
    )
    group.add_argument(
        '--lora_modules_to_save',
        nargs='*',
        type=str,
        help='List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint',
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
