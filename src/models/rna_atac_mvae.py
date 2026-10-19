from typing import Any, Dict, Tuple, List, Optional
from torch import Tensor, nn
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from models.base_vae import BaseVAE
from modules.encoder import Encoder
from modules.decoder import Decoder, NBDecoder
from modules.likelihoods import nb_log_likelihood

class RNA_ATAC_MVAE(BaseVAE):

    def __init__(self, input_dims: List[int], hidden_dims: Optional[List[List[int]]] = None, latent_dim: int = 16, rna_likelihood: str = "gaussian") -> None:
        """
        rna_likelihood: "gaussian" -> MSE on log-normalized RNA (original model)
                        "nb"       -> negative binomial on raw RNA counts
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [[128, 64], [128, 64]]

        assert len(input_dims) == 2
        assert len(hidden_dims) == 2
        assert rna_likelihood in ("gaussian", "nb")

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.rna_likelihood = rna_likelihood

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

        if self.rna_likelihood == "nb":
            self.rna_decoder = NBDecoder(self.rna_input_dim, self.latent_dim, hidden_dims=self.hidden_dims[0][::-1])
        else:
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

    def decode(self, z: Tensor, library: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """library: (batch, 1) observed total RNA counts, required for the NB decoder."""
        out = {"x_hat_atac": self.atac_decoder(z)}

        if self.rna_likelihood == "nb":
            assert library is not None, "NB decoder needs the library size"
            out["rna_mu"], out["rna_log_mu"], out["rna_theta"] = self.rna_decoder(z, library)
        else:
            out["x_hat_rna"] = self.rna_decoder(z)

        return out

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)

        return mu + epsilon * std

    def forward(self, x_rna: Tensor, x_atac: Tensor, x_rna_counts: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """
        x_rna:        log-normalized RNA, encoder input
        x_atac:       ATAC features, encoder input and reconstruction target
        x_rna_counts: raw RNA counts, reconstruction target when rna_likelihood == "nb"
        """
        mu, logvar = self.encode(x_rna, x_atac)
        z = self.reparameterize(mu, logvar)

        library = None
        if self.rna_likelihood == "nb":
            assert x_rna_counts is not None, "NB likelihood needs raw counts"
            library = x_rna_counts.sum(dim=1, keepdim=True)

        out = self.decode(z, library)
        out.update({"x_rna": x_rna, "x_atac": x_atac, "x_rna_counts": x_rna_counts, "mu": mu, "logvar": logvar})

        return out

    def sample(self, batch_size: int, current_device, library: Optional[Tensor] = None, **kwargs) -> Dict[str, Tensor]:
        z = torch.randn(batch_size, self.latent_dim, device=current_device)
        return self.decode(z, library)

    def reconstruct(self, x_rna: Tensor, x_atac: Tensor, x_rna_counts: Optional[Tensor] = None, **kwargs) -> Tuple[Tensor, Tensor]:
        """Returns (RNA reconstruction, ATAC reconstruction). For NB, the RNA reconstruction is the mean mu."""
        out = self.forward(x_rna, x_atac, x_rna_counts)
        x_hat_rna = out["rna_mu"] if self.rna_likelihood == "nb" else out["x_hat_rna"]
        return x_hat_rna, out["x_hat_atac"]

    def loss_function(self, outputs: Dict[str, Tensor], beta: float = 0.1, lambda_rna: float = 0.5, lambda_atac: float = 0.5, **kwargs):
        mu, logvar = outputs["mu"], outputs["logvar"]

        if self.rna_likelihood == "nb":
            # -log p(x | z), summed over genes, averaged over cells
            recon_rna = -nb_log_likelihood(outputs["x_rna_counts"], outputs["rna_log_mu"], outputs["rna_theta"]).sum(dim=1).mean()
        else:
            recon_rna = F.mse_loss(outputs["x_rna"], outputs["x_hat_rna"], reduction="mean")

        recon_atac = F.mse_loss(outputs["x_atac"], outputs["x_hat_atac"], reduction="mean")

        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        loss = lambda_rna * recon_rna + lambda_atac * recon_atac + beta * kl_loss

        return loss, recon_rna, recon_atac, kl_loss
