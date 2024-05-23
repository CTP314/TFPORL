import torch.nn as nn
import torchkit.pytorch_utils as ptu
from policies.seq_models.mamba_encoder import MambaTrajEncoder

class Mamba(nn.Module):
    name = 'mamba'
    
    def __init__(
        self,
        input_size,
        hidden_size,
        n_layer,
        max_seq_length,  
        **kwargs
    ):
        super().__init__()
        
        self.mamba = MambaTrajEncoder(
            tstep_dim=input_size,
            max_seq_len=max_seq_length,
            horizon=max_seq_length,
            d_model=hidden_size,
            n_layers=n_layer,
        )
        self.hidden_size = hidden_size
        
    def get_zero_internal_state(self, batch_size=None):
        return None
    
    
    def forward(self, input_embeds, h_0):
        input_embeds = input_embeds.permute(1, 0, 2)
        ouputs, hidden_state = self.mamba(input_embeds, hidden_state=None)
        return ouputs.permute(1, 0, 2), hidden_state