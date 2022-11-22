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
    def __init__(self, observation_shape=(28,28), z_dim=32, hidden_dim=32):
        super(VariationalEncoder, self).__init__()

        self.observation_shape = observation_shape

        # Define layers
        self.net = torch.nn.Sequential(
            # BLOCK 1
            torch.nn.Conv2d(1, hidden_dim, kernel_size=7, padding=1),               # (B x hidden_dim x 24 x 24)
            torch.nn.BatchNorm2d(hidden_dim),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU(),

            # BLOCK 2
            torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2),      # (B x hidden_dim x 24 x 24)
            torch.nn.BatchNorm2d(hidden_dim),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU(),

            # BLOCK 3
            torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2),      # (B x hidden_dim x 24 x 24)
            torch.nn.BatchNorm2d(hidden_dim),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU(),

            # BLOCK 4
            torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2),      # (B x hidden_dim x 24 x 24)
            torch.nn.BatchNorm2d(hidden_dim),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU(),

            # BLOCK 5
            torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2),      # (B x hidden_dim x 24 x 24)
            torch.nn.BatchNorm2d(hidden_dim),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU(),

            # BLOCK 6
            torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=2),      # (B x hidden_dim x 26 x 26)
            torch.nn.BatchNorm2d(hidden_dim),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU(),

            # FINAL LAYER
            torch.nn.Conv2d(hidden_dim, 2*z_dim, kernel_size=1, padding=0),         # (B x 2 * z_dim x 26 x 26)
        )

    def forward(self, x):
        x = x.view(x.shape[0], 1, *self.observation_shape)      # (B x 1 x 28 x 28)

        # Get latent parameters
        z_params = self.net(x)
        # Average on spatial dimensions (as stated in the paper)
        z_params = z_params.mean(dim=(2, 3))                                  # (B x 2 * zdim)
        return z_params.chunk(2, dim=-1)

class Decoder(torch.nn.Module):
    def __init__(self, observation_shape=((28, 28)), z_dim=32, hidden_dim=128, num_vals=256):
        super(Decoder, self).__init__()

        self.num_vals = num_vals
        self.observation_shape = observation_shape

        self.C1 = torch.nn.Sequential(
            torch.nn.Conv2d(z_dim, hidden_dim, kernel_size=1, padding=0,),
            torch.nn.BatchNorm2d(hidden_dim),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU(),)

        self.C2 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1,),
            torch.nn.BatchNorm2d(hidden_dim),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU(),)

        self.C3 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, ),
            torch.nn.BatchNorm2d(hidden_dim),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU(),)

        self.C4 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2, ),
            torch.nn.BatchNorm2d(hidden_dim),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU(),)

        self.C5 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2, ),
            torch.nn.BatchNorm2d(hidden_dim),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU(),)

        self.C6 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_dim, 1, kernel_size=1, padding=0),)

    def forward(self, z):
        # expand to 2 x 2 x z_dim as they mention in the paper (formatted correctly to torch.Conv2D
        z = z.unsqueeze(-1).unsqueeze(-1)
        x_hat = z.expand(-1, -1, 2, 2) # (B x C x 2 x 2)

        x_hat = self.C1(x_hat)
        x_hat = torch.nn.functional.interpolate(x_hat, mode='bilinear', scale_factor=2) # (B x hidden_dim x 4 x 4)
        x_hat = self.C2(x_hat)
        x_hat = torch.nn.functional.interpolate(x_hat, mode='bilinear', scale_factor=2) # (B x hidden_dim x 8 x 8)
        x_hat = self.C3(x_hat)
        x_hat = torch.nn.functional.interpolate(x_hat, mode='bilinear', scale_factor=2) # (B x hidden_dim x 16 x 16)
        x_hat = self.C4(x_hat)
        x_hat = torch.nn.functional.interpolate(x_hat, mode='bilinear', scale_factor=2) # (B x hidden_dim x 32 x 32)
        x_hat = self.C5(x_hat)
        x_hat = self.C6(x_hat)

        return x_hat[:, :, 2:30, 2:30]

class VariationalAutoEncoder(torch.nn.Module):
    def __init__(self, observation_shape=(28,28), z_dim=32):
        super(VariationalAutoEncoder, self).__init__()
        self.observation_shape = observation_shape

        # Setup model components
        self.encoder = VariationalEncoder(observation_shape, z_dim)
        self.decoder = Decoder(observation_shape, z_dim)
        self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2*z_dim]))) # standard gaussian

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
        x_hat = self.decoder(z)
        return Bernoulli(logits=x_hat)

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
        log_pz = reduce(pz.log_prob(z))
        log_qz = reduce(qz.log_prob(z))
        log_px = reduce(px.log_prob(x.view(x.shape[0], -1, *model.observation_shape)))

        # compute the elbo
        kl = log_qz - log_pz
        elbo = log_px - kl
        beta_elbo = log_px - self.beta * kl

        # define the loss function
        loss = -beta_elbo.mean() # or beta_elbo

        # setup diagnostics of the inference
        with torch.no_grad():
            diagnostics = {'elbo': elbo, 'log_px': log_px, 'kl': kl}

        return loss, diagnostics, outputs