import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Any, Optional, List, Tuple

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

class CountDecoder(nn.Module):
    """Decoder for count data. 
    
    Predicts per cell feature proportions rho(z) and scales them by observed library size 
    to obtain the mean of a distribution (NB or Poisson) mu = l * rho for each cell and feature."""
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

        self.hidden = nn.Sequential(*modules)
        self.fc_rho = nn.Linear(self.hidden_dims[-1], self.input_dim)

    def forward(self, z: Tensor, library: Tensor) -> Tuple[Tensor, Tensor]:
        logits = self.fc_rho(self.hidden(z))
        log_rho = F.log_softmax(logits, dim=-1)
        log_mu = log_rho + torch.log(library)
        mu = torch.exp(log_mu)

        return mu, log_mu


class NBDecoder(CountDecoder):
    """Decoder for negative binomial likelihood on raw counts.

    Inherits the NB mean mu = l * rho from CountDecoder. The inverse
    dispersion theta is a free parameter per gene, shared across cells.
    """
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: Optional[List[int]]):
        super().__init__(input_dim, latent_dim, hidden_dims)

        # Unconstrained parameter, theta = exp(log_theta) > 0
        self.log_theta = nn.Parameter(torch.randn(self.input_dim))

    def forward(self, z: Tensor, library: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        z:       (batch, latent_dim)
        library: (batch, 1) total counts per cell
        returns mu, log_mu: (batch, n_genes) and theta: (n_genes,)
        """
        mu, log_mu = super().forward(z, library)
        theta = torch.exp(self.log_theta)

        return mu, log_mu, theta

class PoissonDecoder(CountDecoder):
    """Decoder for Poisson likelihood on fragment counts.
    
    Inherits the Poisson mean mu = l * rho from CountDecoder."""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: Optional[List[int]]):
        super().__init__(input_dim, latent_dim, hidden_dims)

    def forward(self, z: Tensor, library: Tensor) -> Tuple[Tensor, Tensor]:
        """
        z:       (batch, latent_dim)
        library: (batch, 1) total counts per cell
        returns mu, log_mu: (batch, n_genes)
        """
        return super().forward(z, library)