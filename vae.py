from typing import *

import torch
from torch.distributions import Distribution
from torch import Tensor

import numpy as np

class ReparameterizedDiagonalGaussian(Distribution):
    """
    A distribution `N(y | mu, sigma I)` compatible with the reparameterization trick given `epsilon ~ N(0, 1)`.
    """

    def __init__(self, mu: Tensor, log_sigma: Tensor):
        assert mu.shape == log_sigma.shape, f"Tensors `mu` : {mu.shape} and ` log_sigma` : {log_sigma.shape} must be of the same shape"
        self.mu = mu
        self.sigma = log_sigma.exp()

    def sample_epsilon(self) -> Tensor:
        """`\eps ~ N(0, I)`"""
        return torch.empty_like(self.mu).normal_()

    def sample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (without gradients)"""
        with torch.no_grad():
            return self.rsample()

    def rsample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (with the reparameterization trick) """
        return self.mu + self.sigma * self.sample_epsilon()

    def log_prob(self, z: Tensor) -> Tensor:
        """return the log probability: log `p(z)`"""
        return torch.distributions.Normal(self.mu, self.sigma).log_prob(z)


class VariationalEncoder(torch.nn.Module):
    def __init__(self, observation_shape=((28,28)), z_dim=32, hidden_dim=128):
        super(VariationalEncoder, self).__init__()

        # Define layers
        self.net = torch.nn.Sequential(
            torch.nn.Linear(np.prod(observation_shape), hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 2*z_dim)
        )

    def forward(self, x):
        # Parameters for the
        z_params = self.net(x.flatten())
        return z_params.chunk(2, dim=-1)

class Decoder(torch.nn.Module):
    def __init__(self, z_dim=32, observation_shape=((28, 28)), hidden_dim=128):
        super(Decoder, self).__init__()

        self.observation_shape = observation_shape
        self.net = torch.nn.Sequential(
            torch.nn.Linear(z_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, np.prod(self.observation_shape))
        )

    def forward(self, z):
        x_ = self.net(z)
        return x_.reshape((-1, 1) + self.observation_shape)


class VariationalAutoEncoder(torch.nn.Module):
    def __init__(self, observation_shape=(28,28), z_dim=32):
        super(VariationalAutoEncoder, self).__init__()

        # TODO: self.prior_params

    def posterior(self, x: Tensor):
        pass

    def prior(self, batch_size: int=1) -> Distribution:
        pass

    def observation_model(self, z:Tensor) -> Distribution:
        pass

    def forward(self, x) -> Dict[str, Any]:
        pass

    def sample_from_prior(self, batch_size: int = 100):
        pass





if __name__ == '__main__':


    encoder = VariationalEncoder()
    x = torch.zeros((28, 28))
    encoder(x)
    vae = VAE()