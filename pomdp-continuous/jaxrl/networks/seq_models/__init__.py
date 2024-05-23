from .recurrence import VanillaLSTM
from .linear_recurrence import VanillaLRU
from .gpt2_transformer import TransformerGPT


def get_seq_model(name):
    if name == "lstm":
        return VanillaLSTM
    elif name == "gpt":
        return TransformerGPT
    elif name == "lru":
        return VanillaLRU
    else:
        raise ValueError
