from .lru import LRUModel
import torch.nn as nn
import torchkit.pytorch_utils as ptu
import torch
import numpy as np
import math
from types import SimpleNamespace

class LRU(nn.Module):
    name = 'lru'
    
    def __init__(self, input_size, hidden_size, n_layer, drop, gating, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        assert input_size == hidden_size
        self.args = SimpleNamespace(
            **dict(
                bert_hidden_units=hidden_size,
                bert_attn_dropout=drop,
                bert_dropout=drop,
                bert_num_blocks=n_layer,
                gating=gating,
            )
        )
        self.lru = LRUModel(self.args)
        self.truncated_normal_init()
    
    def forward(self, input, h0=None):
        if h0 is None:
            h0 = self.get_zero_internal_state(input.size(0))
        out, h = self.lru(input.permute(1, 0, 2), h0)
        return out.permute(1, 0, 2), h
    
    def get_zero_internal_state(self, batch_size=None):
        return [None] * self.args.bert_num_blocks
    
    def truncated_normal_init(self, mean=0, std=0.02, lower=-0.04, upper=0.04):
        with torch.no_grad():
            l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
            u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.

            for n, p in self.named_parameters():
                if not 'layer_norm' in n and 'params_log' not in n:
                    if torch.is_complex(p):
                        p.real.uniform_(2 * l - 1, 2 * u - 1)
                        p.imag.uniform_(2 * l - 1, 2 * u - 1)
                        p.real.erfinv_()
                        p.imag.erfinv_()
                        p.real.mul_(std * math.sqrt(2.))
                        p.imag.mul_(std * math.sqrt(2.))
                        p.real.add_(mean)
                        p.imag.add_(mean)
                    else:
                        p.uniform_(2 * l - 1, 2 * u - 1)
                        p.erfinv_()
                        p.mul_(std * math.sqrt(2.))
                        p.add_(mean)
