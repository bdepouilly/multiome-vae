from typing import Any, Dict, Tuple, List, Optional
from torch import Tensor, nn
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from models.base_vae import BaseVAE
from modules.encoder import Encoder
from modules.decoder import Decoder, NBDecoder, PoissonDecoder
from modules.likelihoods import nb_log_likelihood, poisson_log_likelihood

class RNA_ATAC_MVAE(BaseVAE):

    def __init__(self, input_dims: List[int], hidden_dims: Optional[List[List[int]]] = None, latent_dim: int = 16, rna_likelihood: str = "gaussian", atac_likelihood: str = "gaussian", n_peaks: Optional[int] = None) -> None:
        """
        input_dims:      [n_genes, n_atac_features], dimensions of the encoder inputs
        rna_likelihood:  "gaussian" -> MSE on log-normalized RNA (original model)
                         "nb"       -> negative binomial on raw RNA counts
        atac_likelihood: "gaussian" -> MSE on the ATAC features (LSI) used as encoder input
                         "poisson"  -> Poisson on raw fragment counts per peak
        n_peaks:         number of peaks, i.e. the ATAC decoder output size; required for "poisson",
                         since it differs from the encoder input size input_dims[1]
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [[128, 64], [128, 64]]

        assert len(input_dims) == 2
        assert len(hidden_dims) == 2
        assert rna_likelihood in ("gaussian", "nb")
        assert atac_likelihood in ("gaussian", "poisson")
        if atac_likelihood == "poisson":
            assert n_peaks is not None

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.rna_likelihood = rna_likelihood
        self.atac_likelihood = atac_likelihood
        self.n_peaks = n_peaks

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
        if self.atac_likelihood == "poisson":
            self.atac_decoder = PoissonDecoder(self.n_peaks, self.latent_dim, hidden_dims=self.hidden_dims[1][::-1])
        else:
            self.atac_decoder = Decoder(self.atac_input_dim, self.latent_dim, hidden_dims=self.hidden_dims[1][::-1])


    def encode(self, x_rna: Tensor, x_atac: Tensor) -> Tuple[Tensor, Tensor]:
        enc_rna = self.rna_encoder(x_rna)
        enc_atac = self.atac_encoder(x_atac)

        enc = torch.cat([enc_rna, enc_atac], dim=1)

        mu = self.fc_mu(enc)
        logvar = self.fc_logvar(enc)

        return (mu, logvar)

    def decode(self, z: Tensor, rna_library: Optional[Tensor] = None, atac_library: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """
        rna_library:  (batch, 1) observed total RNA counts, required for the NB decoder.
        atac_library: (batch, 1) observed total fragment counts, required for the poisson decoder.
        """
        out = {}

        if self.rna_likelihood == "nb":
            assert rna_library is not None, "NB decoder needs the library size"
            out["rna_mu"], out["rna_log_mu"], out["rna_theta"] = self.rna_decoder(z, rna_library)
        else:
            out["x_hat_rna"] = self.rna_decoder(z)

        if self.atac_likelihood == "poisson":
            assert atac_library is not None, "Poisson decoder needs the library size"
            out["atac_mu"], out["atac_log_mu"] = self.atac_decoder(z, atac_library)
        else:
            out["x_hat_atac"] = self.atac_decoder(z)
        
        return out

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)

        return mu + epsilon * std

    def forward(self, x_rna: Tensor, x_atac: Tensor, x_rna_counts: Optional[Tensor] = None, x_atac_counts: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """
        x_rna:         log-normalized RNA, encoder input
        x_atac:        ATAC features (LSI), encoder input; also the reconstruction target when atac_likelihood == "gaussian"
        x_rna_counts:  raw RNA counts, reconstruction target when rna_likelihood == "nb"
        x_atac_counts: raw fragment counts, reconstruction target when atac_likelihood == "poisson"
        """
        mu, logvar = self.encode(x_rna, x_atac)
        z = self.reparameterize(mu, logvar)

        rna_library = None
        if self.rna_likelihood == "nb":
            assert x_rna_counts is not None, "NB likelihood needs raw counts"
            rna_library = x_rna_counts.sum(dim=1, keepdim=True)
        
        atac_library = None
        if self.atac_likelihood == "poisson":
            assert x_atac_counts is not None, "Poisson likelihood needs raw counts"
            atac_library = x_atac_counts.sum(dim=1, keepdim=True)

        out = self.decode(z, rna_library, atac_library)
        out.update({"x_rna": x_rna, "x_atac": x_atac, "x_rna_counts": x_rna_counts, "x_atac_counts": x_atac_counts, "mu": mu, "logvar": logvar})

        return out

    def sample(self, batch_size: int, current_device, rna_library: Optional[Tensor] = None, atac_library: Optional[Tensor] = None, **kwargs) -> Dict[str, Tensor]:
        z = torch.randn(batch_size, self.latent_dim, device=current_device)
        return self.decode(z, rna_library, atac_library)

    def reconstruct(self, x_rna: Tensor, x_atac: Tensor, x_rna_counts: Optional[Tensor] = None, x_atac_counts: Optional[Tensor] = None, **kwargs) -> Tuple[Tensor, Tensor]:
        """Returns (RNA reconstruction, ATAC reconstruction). For NB/Poisson, the RNA/ATAC reconstruction is the mean mu."""
        out = self.forward(x_rna, x_atac, x_rna_counts, x_atac_counts)
        x_hat_rna = out["rna_mu"] if self.rna_likelihood == "nb" else out["x_hat_rna"]
        x_hat_atac = out["atac_mu"] if self.atac_likelihood == "poisson" else out["x_hat_atac"]
        return x_hat_rna, x_hat_atac

    def loss_function(self, outputs: Dict[str, Tensor], beta: float = 0.1, lambda_rna: float = 0.5, lambda_atac: float = 0.5, **kwargs):
        mu, logvar = outputs["mu"], outputs["logvar"]

        if self.rna_likelihood == "nb":
            # -log p(x | z), summed over genes, averaged over cells
            recon_rna = -nb_log_likelihood(outputs["x_rna_counts"], outputs["rna_log_mu"], outputs["rna_theta"]).sum(dim=1).mean()
        else:
            recon_rna = F.mse_loss(outputs["x_rna"], outputs["x_hat_rna"], reduction="mean")

        if self.atac_likelihood == "poisson":
            recon_atac = -poisson_log_likelihood(outputs["x_atac_counts"], outputs["atac_log_mu"]).sum(dim=1).mean()
        else:
            recon_atac = F.mse_loss(outputs["x_atac"], outputs["x_hat_atac"], reduction="mean")

        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        loss = lambda_rna * recon_rna + lambda_atac * recon_atac + beta * kl_loss

        return loss, recon_rna, recon_atac, kl_loss
