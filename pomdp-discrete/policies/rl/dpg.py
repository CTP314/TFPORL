from typing import Any, Tuple
import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from .base import RLAlgorithmBase
from policies.models.actor import CategoricalPolicy
from torchkit.networks import FlattenMlp
import torchkit.pytorch_utils as ptu

class DPG(RLAlgorithmBase):
    name = "dpg"
    continuous_action = False
    use_target_actor = False
    
    @staticmethod
    def build_actor(input_size, action_dim, hidden_sizes, **kwargs):
        return CategoricalPolicy(
            obs_dim=input_size,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            **kwargs,
        )
        
    def select_action(self, actor, observ, deterministic: bool) -> Any:
        return actor(observ, deterministic, return_log_prob=False)[0]
    
    @staticmethod
    def forward_actor(actor, observ) -> Tuple[Any, Any]:
        _, probs, log_probs = actor(observ, return_log_prob=True)
        return probs, log_probs
    
    def actor_loss(
        self,
        markov_actor: bool,
        actor,
        observs,
        rtg,
        actions=None,
        rewards=None,
    ):
        if markov_actor:
            new_probs, log_probs = self.forward_actor(actor, observs)
        else:
            new_probs, log_probs = actor(
                prev_actions=actions[:-1], rewards=rewards[:-1], observs=observs[:-1]
            )
        
        stored_action = actions[1:]
        log_probs_pred = log_probs.gather(dim=-1, index=stored_action)
        policy_loss = rtg * log_probs
        
        return policy_loss, log_probs_pred
