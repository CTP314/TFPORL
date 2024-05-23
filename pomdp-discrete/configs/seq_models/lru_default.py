from ml_collections import ConfigDict
from typing import Tuple
from configs.seq_models.name_fns import name_fn

def lru_name_fn(config: ConfigDict, max_episode_steps: int) -> Tuple[ConfigDict, str]:
    config, name = name_fn(config, max_episode_steps)

    config.model.seq_model_config.hidden_size = 0
    if config.model.observ_embedder is not None:
        config.model.seq_model_config.hidden_size += (
            config.model.observ_embedder.hidden_size
        )
    if config.model.action_embedder is not None:
        config.model.seq_model_config.hidden_size += (
            config.model.action_embedder.hidden_size
        )
    if config.model.reward_embedder is not None:
        config.model.seq_model_config.hidden_size += (
            config.model.reward_embedder.hidden_size
        )

    config.model.seq_model_config.max_seq_length = (
        config.sampled_seq_len + 1
    )  # NOTE: zero-prepend

    return config, name

def get_config():
    config = ConfigDict()
    config.name_fn = lru_name_fn

    config.is_markov = False
    config.is_attn = False
    config.use_dropout = False

    config.sampled_seq_len = -1

    config.clip = False
    config.max_norm = 1.0
    config.use_l2_norm = False

    # fed into Module
    config.model = ConfigDict()

    # seq_model specific
    config.model.seq_model_config = ConfigDict()
    config.model.seq_model_config.name = "lru"
    config.model.seq_model_config.drop = 0.1
    config.model.seq_model_config.gating = False
    config.model.seq_model_config.n_layer = 2

    # embedders
    config.model.observ_embedder = ConfigDict()
    config.model.observ_embedder.name = "mlp"
    config.model.observ_embedder.hidden_size = 64

    config.model.action_embedder = ConfigDict()
    config.model.action_embedder.name = "mlp"
    config.model.action_embedder.hidden_size = 64

    config.model.reward_embedder = ConfigDict()
    config.model.reward_embedder.name = "mlp"
    config.model.reward_embedder.hidden_size = 0

    return config
