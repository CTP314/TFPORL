import torch.nn as nn
import torchkit.pytorch_utils as ptu
import ding.torch_utils.network.gtrxl as gtrxl

class GTrXL(nn.Module):
    name = 'gtrxl'
    
    def __init__(
        self,
        input_size,
        hidden_size,
        n_layer,
        n_head,
        pdrop,
        max_seq_length,  
        **kwargs
    ):
        super().__init__()
        
        self.gtrxl = gtrxl.GTrXL(
            input_dim=input_size,
            embedding_dim=hidden_size,
            head_num=n_head,
            layer_num=n_layer,
            dropout_ratio=pdrop,
        )
        self.hidden_size = hidden_size
        
    def forward(self, input_embeds, h_0):
        if h_0 is None:
            self.gtrxl.reset_memory()
        
        output = self.gtrxl(input_embeds)
        return output['logit'], output['memory']
    
    def get_zero_internal_state(self, batch_size=None):
        return None