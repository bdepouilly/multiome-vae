import torch
from torch.distributions import Poisson
from modules.likelihoods import nb_log_likelihood, poisson_log_likelihood
from modules.decoder import PoissonDecoder
from scipy.sparse import csr_matrix
import numpy as np

torch.manual_seed(42)
torch.set_default_dtype(torch.float64)

# Inputs 

x = torch.randint(0, 20, (64, 100))
x[:, :20] = 0
x[:, -5:] = 590
log_mu = torch.empty(64, 100).uniform_(-15, 7)

ours = poisson_log_likelihood(x, log_mu)
ref = Poisson(rate=log_mu.exp()).log_prob(x)

err = (ours - ref).abs().max().item()
print(f"[1] vs torch Poisson: max abs diff = {err:.2e}")
assert err < 1e-8, "Poisson log-likelihood disagrees with torch"

for theta in [1e2, 1e4, 1e6, 1e8, 1e10]:
    ours = nb_log_likelihood(x, log_mu, torch.tensor(theta))
    ref = Poisson(rate=log_mu.exp()).log_prob(x)
    err = (ours - ref).abs().max().item()
    print(f"our NB log-lik vs torch Poisson: max abs diff = {err:.2e}")

dec = PoissonDecoder(output_dim=66118, latent_dim=32, hidden_dims=[64, 128])
z = torch.randn(8, 32)
library = torch.randint(100, 70000, (8,1))
mu, log_mu_dec = dec(z, library)
assert mu.shape == (8, 66118), "Output shape doesn't match expected"
row_sums = mu.sum(dim=1, keepdim=True)
err = (row_sums - library).abs().max().item()
print(f"Row sum of mu vs library: max abs diff = {err:.2e}")
assert err < 1e-8
assert torch.allclose(log_mu_dec.exp(), mu, rtol=1e-5), "log_mu inconsistent with mu"
print("Decoder: shapes OK, row sums match library (max rel err "
      f"{((row_sums - library).abs() / library).max().item():.1e})")
