
class SpecialToken:
    SOS = '<s>'
    EOS = '</s>'
    UNK = '<unk>'
    PAD = EOS  # as llama does not use padding token
    START_INST = '[INST]'
    END_INST = '[/INST]'
    START_SYS = '<<SYS>>'
    END_SYS = '<</SYS>>'


SYSTEM_PROMPT = 'You are a doctor. You are talking to a patient. You are trying to diagnose the patient\'s illness. Please answer the following patent\'s question.'
