from typing import *

import torch
from torch.distributions import Distribution, Bernoulli, Categorical, constraints
from torch import Tensor

import numpy as np

# Parts of the code are heavily inspired / directly taken from the course 02456 Deep Learning at DTU.


class ReparameterizedDiagonalGaussian:
    """
    A distribution `N(y | mu, sigma I)` compatible with the reparameterization trick given `epsilon ~ N(0, 1)`.
    """

    def __init__(self, mu: Tensor, log_sigma: Tensor):
        assert mu.shape == log_sigma.shape, f"Tensors `mu` : {mu.shape} and ` log_sigma` : {log_sigma.shape} must be of the same shape"
        arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
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
    def __init__(self, observation_shape=(28,28), z_dim=32, hidden_dim=128):
        super(VariationalEncoder, self).__init__()

        # Define layers
        self.net = torch.nn.Sequential(
            torch.nn.Linear(784, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 2*z_dim),
        )

    def forward(self, x):
        # Parameters for the
        z_params = self.net(x)
        return z_params.chunk(2, dim=-1)

class Decoder(torch.nn.Module):
    def __init__(self, observation_shape=((28, 28)), z_dim=32, hidden_dim=128, num_vals=256):
        super(Decoder, self).__init__()

        self.num_vals = num_vals
        self.observation_shape = observation_shape
        self.net = torch.nn.Sequential(
            torch.nn.Linear(z_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, np.prod(self.observation_shape) * num_vals),
            torch.nn.Softmax(dim=-1),
        )

    def forward(self, z):
        logits = self.net(z)
        return logits.view(logits.shape[0], *self.observation_shape, self.num_vals)

class VariationalAutoEncoder(torch.nn.Module):
    def __init__(self, observation_shape=(28,28), z_dim=32):
        super(VariationalAutoEncoder, self).__init__()
        self.observation_shape = observation_shape

        # Setup model components
        self.encoder = VariationalEncoder(observation_shape, z_dim)
        self.decoder = Decoder(observation_shape, z_dim)
        self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2*z_dim]))) # TODO: why is this?

    def posterior(self, x: Tensor):
        # Get latent parameters
        posterior_params = self.encoder(x)
        mu, log_sigma = posterior_params
        return ReparameterizedDiagonalGaussian(mu, log_sigma)

    def prior(self, batch_size: int=1) -> Distribution:
        prior_params = self.prior_params.expand(batch_size, *self.prior_params.shape[-1:])
        mu, log_sigma = prior_params.chunk(2, dim=-1)
        return ReparameterizedDiagonalGaussian(mu, log_sigma)

    def observation_model(self, z: Tensor) -> Distribution:
        px_logits = self.decoder(z)
        return Categorical(px_logits)

    def forward(self, x) -> Dict[str, Any]:
        # change data shape
        x = x.view(x.size(0), -1)

        # encode x into the posterior
        qz = self.posterior(x)

        # define prior distribution
        pz = self.prior(batch_size=x.size(0))

        # sample from the prior
        z = qz.rsample()

        # observation model
        px = self.observation_model(z)
        return {'px': px, 'pz': pz, 'qz': qz, 'z': z}

    def sample_from_prior(self, batch_size: int = 100):

        # setup prior
        pz = self.prior(batch_size=batch_size)

        # sample from the prior
        z = pz.rsample()

        # observation model
        px = self.observation_model(z)
        return {'px': px, 'pz': pz, 'z': z}



def reduce(x:Tensor) -> Tensor:
    """for each datapoint: sum over all dimensions"""
    return x.view(x.size(0), -1).sum(dim=1)

class VariationalInference(torch.nn.Module):
    def __init__(self, beta: float = 1.):
        super().__init__()
        self.beta = beta

    def forward(self, model: torch.nn.Module, x: Tensor) -> Tuple[Tensor, Dict]:
        # forward pass throught the VAE model
        outputs = model(x)

        # unpack outputs
        px, pz, qz, z = [outputs[k] for k in ["px", "pz", "qz", "z"]]

        # get log-probs to be used in kl and elbo calculation
        log_px = reduce(px.log_prob(x.view(-1, *model.observation_shape) * 256))
        log_pz = reduce(pz.log_prob(z))
        log_qz = reduce(qz.log_prob(z))

        # compute the elbo
        kl = log_qz - log_pz
        elbo = log_px - kl
        beta_elbo = log_px - self.beta * kl

        # define the loss function
        loss = -beta_elbo.mean()

        # setup diagnostics of the inference
        with torch.no_grad():
            diagnostics = {'elbo': elbo, 'log_px': log_px, 'kl': kl}

        return loss, diagnostics, outputs