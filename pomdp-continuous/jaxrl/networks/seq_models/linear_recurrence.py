import functools
from typing import Any, Callable, Optional, Tuple

from flax import linen as nn
import jax
from jax import numpy as jnp
from jaxrl.networks.initializer import default_kernel_init, default_bias_init
from .base import SequenceModelBase
from .minimal_lru import LRU, StackedEncoderModel, BatchStackedEncoderModel

class VanillaLRU(SequenceModelBase):
    hidden_size: int
    n_layer: int = 2
    pdrop: float = 0.1

    def setup(self):
        lru = functools.partial(LRU, d_hidden=self.hidden_size, d_model=self.hidden_size)
        self.lru_module = BatchStackedEncoderModel(lru, self.hidden_size, self.n_layer, dropout=self.pdrop)

    def __call__(self, carry, x, rng=None):
        if self.has_variable("params", "lru_module"):
            lru_params = {"params": self.variables["params"]["lru_module"]}

            if rng is None:  # evaluation
                deterministic, dropout_rng = True, None
            else:  # training or exploration
                # https://flax.readthedocs.io/en/latest/api_reference/flax.errors.html#flax.errors.InvalidRngError
                # Module.apply() has rng for dropout
                deterministic, dropout_rng = False, {"dropout": rng}

            out = self.lru_module.apply(
                lru_params,
                carry,
                x,
                rngs=dropout_rng,
            )

            return out
        else:
            # init
            return self.lru_module(
                carry,
                x,
            )
    def forward(self, embedded_inputs, initial_states, **kwargs):
        hidden_states, outputs = self.__call__(initial_states, embedded_inputs, **kwargs)

        # ((B, T, D), (B, T, D)), (B, T, D)
        return (hidden_states, outputs), None

    def forward_per_step(self, embedded_inputs, initial_states, **kwargs):
        hidden_states, outputs = self.__call__(initial_states, embedded_inputs, **kwargs)

        # ((D), (D)), (D)
        return (hidden_states, outputs), None
    
    def initialize_carry(self, batch_dims):
        batch_size = batch_dims[0] if len(batch_dims) == 1 else 1
        return [None] * self.n_layer