import torch
from models.base_vae import BaseVAE
from torch import nn
from torch import Tensor
from typing import List, Any, Optional
import torch.nn.functional as F

class RNA_VAE(BaseVAE):
    
    def __init__(self, input_dim: int = 5000, hidden_dims: Optional[List[int]] = None, latent_dim: int = 16) -> None:
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 128]
            
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        
        #Encoder
        
        modules = []
        
        in_dim = input_dim
            
        for h in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_dim, h),
                    nn.LeakyReLU()
                )
            )
            in_dim = h
            
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        #Decoder
        
        modules = []
        
        l_dim = latent_dim
        
        rev_hidden_dims = hidden_dims[::-1]
        
        for h in rev_hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(l_dim, h),
                    nn.LeakyReLU()
                )
            )
            l_dim = h
            
        last_layer = nn.Linear(rev_hidden_dims[-1], input_dim)
            
        self.decoder = nn.Sequential(*modules, last_layer)
        
    def encode(self, input: Tensor) -> List[Tensor]:
            
        result = self.encoder(input)
        
        mu = self.fc_mu(result)
        logvar = self.fc_logvar(result)
        
        return [mu, logvar]
    
    def decode(self, input: Tensor) -> Tensor:
        
        result = self.decoder(input)
        
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        
        return mu + epsilon * std

    def forward(self, x: Tensor) -> List[Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return [x_hat, x, mu, logvar]
    
    def sample(self, batch_size: int, current_device, **kwargs) -> Tensor:
        z = torch.randn(batch_size, self.latent_dim, device=current_device)
        return self.decode(z)
    
    def reconstruct(self, x: Tensor) -> Tensor:
        return self.forward(x)[0]
    
    def loss_function(self, *inputs: Any, beta: float = 1.0, **kwargs) -> List[Tensor]:
        x_hat, x, mu, logvar = inputs
        
        recon_loss = F.mse_loss(x_hat, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        
        loss = recon_loss + beta * kl_loss
        
        return loss, recon_loss, kl_loss
        