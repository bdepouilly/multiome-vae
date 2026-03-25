from typing import Any, Tuple, List, Optional
from torch import Tensor, nn
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from models.base_vae import BaseVAE
from modules.encoder import Encoder
from modules.decoder import Decoder

class RNA_ATAC_MVAE(BaseVAE):
    
    def __init__(self, input_dims: List[int], hidden_dims: Optional[List[List[int]]] = None, latent_dim: int = 16) -> None:
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [[128, 64], [128, 64]]
            
        assert len(input_dims) == 2
        assert len(hidden_dims) == 2
        
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        
        # RNA encoder
        
        self.rna_input_dim = self.input_dims[0]
        self.rna_hidden_dims = self.hidden_dims[0]
        
        self.rna_encoder = Encoder(input_dim=self.rna_input_dim, hidden_dims=self.rna_hidden_dims)
            
        # ATAC encoder
        
        self.atac_input_dim = self.input_dims[1]
        self.atac_hidden_dims = self.hidden_dims[1]
        
        self.atac_encoder = Encoder(input_dim=self.atac_input_dim, hidden_dims=self.atac_hidden_dims)
        
        # Combine encodings
        self.fc_mu = nn.Linear(self.rna_hidden_dims[-1] + self.atac_hidden_dims[-1], self.latent_dim)
        self.fc_logvar = nn.Linear(self.rna_hidden_dims[-1] + self.atac_hidden_dims[-1], self.latent_dim)
        
        # RNA decoder
        
        self.rna_decoder = Decoder(self.rna_input_dim, self.latent_dim, hidden_dims=self.hidden_dims[0][::-1])
        
        # ATAC decoder
        self.atac_decoder = Decoder(self.atac_input_dim, self.latent_dim, hidden_dims=self.hidden_dims[1][::-1])
        
        
    def encode(self, x_rna: Tensor, x_atac: Tensor) -> Tuple[Tensor, Tensor]:
        enc_rna = self.rna_encoder(x_rna)
        enc_atac = self.atac_encoder(x_atac)
        
        enc = torch.cat([enc_rna, enc_atac], dim=1)
        
        mu = self.fc_mu(enc)
        logvar = self.fc_logvar(enc)
        
        return (mu, logvar)
    
    def decode(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        x_hat_rna = self.rna_decoder(z)
        x_hat_atac = self.atac_decoder(z)
        
        return (x_hat_rna, x_hat_atac)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        
        return mu + epsilon * std
    
    def forward(self, x_rna: Tensor, x_atac: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x_rna, x_atac)
        z = self.reparameterize(mu, logvar)
        x_hat_rna, x_hat_atac = self.decode(z)
        
        return x_rna, x_hat_rna, x_atac, x_hat_atac, mu, logvar
    
    def sample(self, batch_size: int, current_device, **kwargs) -> Tuple[Tensor, Tensor]:
        z = torch.randn(batch_size, self.latent_dim, device=current_device)
        return self.decode(z)
    
    def reconstruct(self, x_rna: Tensor, x_atac: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
        result = self.forward(x_rna, x_atac)
        return result[1], result[3]
    
    def loss_function(self, *inputs: Any, beta: float = 0.1, lambda_rna: float = 0.5, lambda_atac: float = 0.5, **kwargs):
        x_rna, x_hat_rna, x_atac, x_hat_atac, mu, logvar = inputs
        
        recon_rna = F.mse_loss(x_rna, x_hat_rna, reduction="mean")
        recon_atac = F.mse_loss(x_atac, x_hat_atac, reduction="mean")
        
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        
        loss = lambda_rna * recon_rna + lambda_atac * recon_atac + beta * kl_loss
        
        return loss, recon_rna, recon_atac, kl_loss