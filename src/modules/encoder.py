from torch import nn, Tensor
from typing import List, Optional, Tuple

class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Optional[List[int]] = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        modules = []
        in_dim = self.input_dim
        for h in self.hidden_dims:
            modules.append(nn.Sequential(
                    nn.Linear(in_dim, h),
                    nn.LeakyReLU()
                )
            )
            in_dim = h
            
        self.encoder = nn.Sequential(*modules)
        
    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)
        
        
        
    