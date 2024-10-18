
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

ALPACA_SYSTEM_PROMPT = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.'
