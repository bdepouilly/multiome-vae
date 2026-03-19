from torch import nn, Tensor
from typing import Any, Optional, List

class Decoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: Optional[List[int]]):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        if self.hidden_dims is None:
            self.hidden_dims = [64, 128]
            
        modules = []
        l_dim = self.latent_dim
        
        for h in self.hidden_dims:
            modules.append(nn.Sequential(
                nn.Linear(l_dim, h),
                nn.LeakyReLU()
            ))
            l_dim = h
        
        last_layer = nn.Linear(self.hidden_dims[-1], self.input_dim)
        
        self.decoder = nn.Sequential(*modules, last_layer)
    
    def forward(self, z: Tensor):
        return self.decoder(z)