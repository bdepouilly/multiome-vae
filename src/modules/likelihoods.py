import torch
from torch import Tensor


def nb_log_likelihood(x: Tensor, log_mu: Tensor, theta: Tensor) -> Tensor:
    """Element-wise log-likelihood of counts under a negative binomial NB(mu, theta).

    mean = mu, variance = mu + mu^2 / theta

    log p(x) = lgamma(x + theta) - lgamma(theta) - lgamma(x + 1)
               + theta * log(theta / (theta + mu))
               + x * log(mu / (theta + mu))

    x:      (batch, n_genes) raw counts
    log_mu: (batch, n_genes) log of the NB mean
    theta:  (n_genes,) inverse dispersion, broadcast over cells
    returns (batch, n_genes); sum over genes for the per-cell log-likelihood
    """
    log_theta = torch.log(theta)
    log_theta_mu = torch.logaddexp(log_theta, log_mu)  # log(theta + mu)

    combinatorial = torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1)

    return combinatorial + theta * (log_theta - log_theta_mu) + x * (log_mu - log_theta_mu)
