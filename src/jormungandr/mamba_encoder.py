import torch
from mamba_ssm import Mamba

from torch import nn

class MambaEncoder(nn.Module):
    def __init__(self, 
                 model_dimension: int = 16, state_expansion_factor: int = 16):
        super(MambaEncoder, self).__init__()
        self.num_layers = 6
        self.layers = nn.ModuleList([
            Mamba(
                d_model=model_dimension, 
                d_state=state_expansion_factor, 
            ) for _ in range(self.num_layers)
        ])


        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, model_dimension)
        Returns:
            Tensor of shape (batch_size, model_dimension)
        """
        
        for layer in self.layers:
            x = layer(x)
        return x