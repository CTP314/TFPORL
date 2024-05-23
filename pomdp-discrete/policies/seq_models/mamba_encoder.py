from torch import nn
import torch.nn.functional as F
import torch
from abc import ABC, abstractmethod
from typing import Tuple, List

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

class TrajEncoder(nn.Module, ABC):
    def __init__(self, tstep_dim: int, max_seq_len: int, horizon: int):
        super().__init__()
        self.tstep_dim = tstep_dim
        self.max_seq_len = max_seq_len
        self.horzion = horizon

    def reset_hidden_state(self, hidden_state, dones):
        return hidden_state

    def init_hidden_state(self, batch_size: int, device: torch.device):
        return None

    @abstractmethod
    def forward(self, seq: torch.Tensor, time_idxs: torch.Tensor, hidden_state=None):
        pass

    @abstractmethod
    def emb_dim(self):
        pass


class Normalization(nn.Module):
    def __init__(self, method: str, d_model: int):
        super().__init__()
        assert method in ["layer", "batch", "none"]
        if method == "layer":
            self.norm = nn.LayerNorm(d_model)
        elif method == "none":
            self.norm = lambda x: x
        else:
            self.norm = nn.BatchNorm1d(d_model)
        self.method = method

    def forward(self, x):
        if self.method == "batch":
            return self.norm(x.transpose(-1, 1)).transpose(-1, 1)
        return self.norm(x)

class _MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, norm: str):
        super().__init__()
        self.norm = Normalization(norm, d_model)
        self.mamba = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )

    def forward(self, seq):
        return seq + self.mamba(self.norm(seq))

    def step(self, seq, conv_state, ssm_state):
        res, new_conv_state, new_ssm_state = self.mamba.step(
            self.norm(seq), conv_state, ssm_state
        )
        return seq + res, new_conv_state, new_ssm_state


class _MambaHiddenState:
    def __init__(self, conv_states: List[torch.Tensor], ssm_states: List[torch.Tensor]):
        assert len(conv_states) == len(ssm_states)
        self.n_layers = len(conv_states)
        self.conv_states = conv_states
        self.ssm_states = ssm_states

    def reset(self, idxs):
        for i in range(self.n_layers):
            # hidden states are initialized to zero
            self.conv_states[i][idxs] = 0.0
            self.ssm_states[i][idxs] = 0.0

    def __getitem__(self, layer_idx: int):
        assert layer_idx < self.n_layers
        return self.conv_states[layer_idx], self.ssm_states[layer_idx]

    def __setitem__(self, layer_idx: int, conv_ssm: Tuple[torch.Tensor]):
        conv, ssm = conv_ssm
        self.conv_states[layer_idx] = conv
        self.ssm_states[layer_idx] = ssm


class MambaTrajEncoder(TrajEncoder):
    """
    "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", Gu and Dao, 2023.

    https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py
    https://github.com/johnma2006/mamba-minimal/tree/master
    """

    def __init__(
        self,
        tstep_dim: int,
        max_seq_len: int,
        horizon: int,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 3,
        norm: str = "layer",
    ):
        super().__init__(tstep_dim, max_seq_len, horizon)

        assert (
            Mamba is not None
        ), "Missing Mamba installation (pip install amago[mamba])"
        self.inp = nn.Linear(tstep_dim, d_model)

        self.mambas = nn.ModuleList(
            [
                _MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    norm=norm,
                )
                for _ in range(n_layers)
            ]
        )
        self.out_norm = Normalization(norm, d_model)
        self._emb_dim = d_model

    def init_hidden_state(self, batch_size: int, device: torch.device):
        conv_states, ssm_states = [], []
        for mamba_block in self.mambas:
            conv_state, ssm_state = mamba_block.mamba.allocate_inference_cache(
                batch_size, max_seqlen=self.max_seq_len
            )
            conv_states.append(conv_state)
            ssm_states.append(ssm_state)
        return _MambaHiddenState(conv_states, ssm_states)

    def reset_hidden_state(self, hidden_state, dones):
        if hidden_state is None:
            return None
        assert isinstance(hidden_state, _MambaHiddenState)
        hidden_state.reset(idxs=dones)
        return hidden_state

    def forward(self, seq, time_idxs=None, hidden_state=None):
        seq = self.inp(seq)
        if hidden_state is None:
            for mamba in self.mambas:
                seq = mamba(seq)
        else:
            assert not self.training
            assert isinstance(hidden_state, _MambaHiddenState)
            for i, mamba in enumerate(self.mambas):
                conv_state_i, ssm_state_i = hidden_state[i]
                seq, new_conv_state_i, new_ssm_state_i = mamba.step(
                    seq, conv_state=conv_state_i, ssm_state=ssm_state_i
                )
                hidden_state[i] = new_conv_state_i, new_ssm_state_i
        return self.out_norm(seq), hidden_state

    @property
    def emb_dim(self):
        return self._emb_dim