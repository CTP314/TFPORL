import torch.nn as nn
import torchkit.pytorch_utils as ptu

class MLP(nn.Module):
    name = "mlp"
    
    def __init__(self, input_size, hidden_size, **kwargs):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )
        self.hidden_size = hidden_size
        self.num_layers = 1
        
    def forward(self, inputs, h_0):
        """
        inputs: (T, B, input_dim)
        h_0: (num_layers=1, B, hidden_size)
        return
        output: (T, B, hidden_size)
        h_n: (num_layers=1, B, hidden_size), only used in inference
        """
        output = self.model(inputs)
        return output, output
    
    def get_zero_internal_state(self, batch_size=1):
        return ptu.zeros((self.num_layers, batch_size, self.hidden_size)).float()