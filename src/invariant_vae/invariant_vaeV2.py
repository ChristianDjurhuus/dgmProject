import torch
import torch.nn.functional as F
from torch.nn import Module
from e2cnn import gspaces
from e2cnn import nn
from utils import rot_img, get_rotation_matrix, get_batch_norm, get_non_linearity
import math
from torch.distributions import Bernoulli


class Encoder(Module):
    def __init__(self, out_dim, hidden_dim=32):
        super().__init__()
        self.out_dim=out_dim
        self.r2_act = gspaces.Rot2dOnR2(N=-1, maximum_frequency=8)
        in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        self.input_type = in_type

        # convolution 1
        out_scalar_fields = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.trivial_repr])
        out_vector_field = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.irrep(1)])
        out_type = out_scalar_fields + out_vector_field
        batch_norm = get_batch_norm(out_scalar_fields, out_vector_field)
        nonlinearity = get_non_linearity(out_scalar_fields, out_vector_field)

        self.block1 = nn.SequentialModule(
            #nn.MaskModule(in_type, 29, margin=1),
            nn.R2Conv(in_type, out_type, kernel_size=7, padding=1, bias=False),
            batch_norm,
            nonlinearity
        )

        # convolution 2
        in_type = out_type
        out_scalar_fields = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.trivial_repr])
        out_vector_field = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.irrep(1)])
        out_type = out_scalar_fields + out_vector_field
        batch_norm = get_batch_norm(out_scalar_fields, out_vector_field)
        nonlinearity = get_non_linearity(out_scalar_fields, out_vector_field)

        self.block2 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            batch_norm,
            nonlinearity
        )
        self.pool1 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

        # convolution 3
        in_type = out_type
        out_scalar_fields = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.trivial_repr])
        out_vector_field = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.irrep(1)])
        out_type = out_scalar_fields + out_vector_field
        batch_norm = get_batch_norm(out_scalar_fields, out_vector_field)
        nonlinearity = get_non_linearity(out_scalar_fields, out_vector_field)

        self.block3 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            batch_norm,
            nonlinearity
        )

        # convolution 4
        in_type = out_type
        out_scalar_fields = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.trivial_repr])
        out_vector_field = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.irrep(1)])
        out_type = out_scalar_fields + out_vector_field
        batch_norm = get_batch_norm(out_scalar_fields, out_vector_field)
        nonlinearity = get_non_linearity(out_scalar_fields, out_vector_field)

        self.block4 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            batch_norm,
            nonlinearity
        )
        self.pool2 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

        # convolution 5
        in_type = out_type
        out_scalar_fields = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.trivial_repr])
        out_vector_field = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.irrep(1)])
        out_type = out_scalar_fields + out_vector_field
        batch_norm = get_batch_norm(out_scalar_fields, out_vector_field)
        nonlinearity = get_non_linearity(out_scalar_fields, out_vector_field)

        self.block5 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            batch_norm,
            nonlinearity
        )

        # convolution 6
        in_type = out_type
        out_scalar_fields = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.trivial_repr])
        out_vector_field = nn.FieldType(self.r2_act, hidden_dim * [self.r2_act.irrep(1)])
        out_type = out_scalar_fields + out_vector_field
        batch_norm = get_batch_norm(out_scalar_fields, out_vector_field)
        nonlinearity = get_non_linearity(out_scalar_fields, out_vector_field)

        self.block6 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=2, bias=False),
            batch_norm,
            nonlinearity
        )
        # convolution 7 --> out
        # the old output type is the input type to the next layer
        in_type = out_type
        out_scalar_fields = nn.FieldType(self.r2_act, 2*out_dim * [self.r2_act.trivial_repr])
        out_vector_field = nn.FieldType(self.r2_act, 1 * [self.r2_act.irrep(1)])
        out_type = out_scalar_fields + out_vector_field

        self.block7 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=1, padding=0, bias=False),
        )

    def forward(self, x: torch.Tensor):
        #x = x.unsqueeze(1)   #TODO: HVORFOR KOMMENTER DEN HER UD
        #x = torch.nn.functional.pad(x, (0, 1, 0, 1), value=0).unsqueeze(1)
        x = nn.GeometricTensor(x, self.input_type)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        x = x.tensor.mean(dim=(2, 3))
        x_0, x_1 = x[:, :2*self.out_dim], x[:, 2*self.out_dim:]
        mu, log_var = x_0[:,:2*self.out_dim//2], x_0[:,2*self.out_dim//2:]
        return mu, log_var, x_1


class Decoder(Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # convolution 1
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_size, hidden_size, kernel_size=1, padding=0,),
            torch.nn.BatchNorm2d(hidden_size),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU()
        )

        # convolution 2
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(hidden_size),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU()
        )

        # convolution 3
        self.block3 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(hidden_size),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU()
        )

        # convolution 4
        self.block4 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_size, hidden_size, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(hidden_size),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU()
        )

        # convolution 5
        self.block5 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_size, hidden_size, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(hidden_size),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU()
        )

        # convolution 6
        self.block6 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_size, 1, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(-1).unsqueeze(-1)  # [bz, emb_dim, 1, 1]
        x = x.expand(-1, -1, 2, 2)
        #pos_emb = torch.Tensor([[1, 2], [4, 3]]).type_as(x).unsqueeze(0).unsqueeze(0).expand(x.size(0), x.size(1), -1, -1)
        #x = x + pos_emb

        x = self.block1(x)
        x = torch.nn.functional.upsample_bilinear(x, scale_factor=2)
        x = self.block2(x)
        x = torch.nn.functional.upsample_bilinear(x, scale_factor=2)
        x = self.block3(x)
        x = torch.nn.functional.upsample_bilinear(x, scale_factor=2)
        x = self.block4(x)
        x = torch.nn.functional.upsample_bilinear(x, scale_factor=2)
        x = self.block5(x)
        x = self.block6(x)
        x = x[:, :, 2:30, 2:30]
        return x


class VAE(Module):
    def __init__(self, emb_dim, hidden_dim):
        super().__init__()
        self.encoder = Encoder(out_dim=emb_dim)
        self.decoder = Decoder(input_size=emb_dim, hidden_size = hidden_dim)

        # define the parameters of the prior, chosen as p(z) = N(0, I)
        self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2*emb_dim])))
    def posterior(self, x):
        '''
        Computes distribution q(x|x) = N(z| mu_x, sigma_x)
        '''
        mu, log_var, v = self.encoder(x)
        return ReparameterizedDiagonalGaussian(mu, log_var), v

    def prior(self, batch_size):
        '''
        Computes distribution p(z)
        '''
        prior_params = self.prior_params.expand(batch_size, *self.prior_params.shape[-1:])
        mu, log_var = prior_params.chunk(2, dim=-1)
        return ReparameterizedDiagonalGaussian(mu, log_var)
    
    def observation_model(self, z, rot):
        '''
        Computes decoded output p(x|z)
        '''
        px_logits = self.decoder(z)
        rot_px_logits = rot_img(px_logits, rot)
        return Bernoulli(logits=rot_px_logits)  #TODO: maybe reshape
    

    def forward(self, x, do_rot=True):
        #TODO: maybe flatten input

        #define posterior q(z|x) and determine rotation
        qz, v = self.posterior(x)

        #define prior p(z)
        pz = self.prior(batch_size=x.shape[0])

        #Sample posterior using reparameterization trick
        z = qz.rsample()

        #define the rotation matrix
        rot = get_rotation_matrix(v)

        #define the observation model p(x|z)
        px = self.observation_model(z, rot)

        #if do_rot:
        #    px = rot_img(px, rot)

        return {'px': px, 'pz': pz, 'qz': qz, 'z': z, 'rot':rot}

    def sample_prior(self, batch_size):
        '''
        sample z ~ p(z) and returns p(x|z)
        '''
        #define prior
        pz = self.prior(batch_size=batch_size)
        #sample prior
        z = pz.rsample()
        #computing p(x|z)
        px = self.observation_model(z)
        return px

class ReparameterizedDiagonalGaussian():
    def __init__(self, mu, logvar):
        assert mu.shape == logvar.shape, "Dimension mismatch mu: {mu.shape}, sigma: {logvar.shape}"
        self.mu = mu
        self.sigma = torch.exp(0.5 * logvar)

    def sample_epsilon(self,):
        '''
        epsilon ~ N(0, 1)
        '''
        return torch.empty_like(self.mu).normal_()
    
    def sample(self,):
        '''
        return sample z ~ N(z| mu, sigma)
        '''
        with torch.no_grad:
            return self.rsample()
        
    def rsample(self,):
        '''
        Return sample `z ~ N(z | mu, sigma)` (with the reparameterization trick)
        '''
        return self.mu + (self.sigma * self.sample_epsilon())
    
    def log_prob(self, z):
        '''
        Returns the log_prob(z)
        Determined by computing the log of the pdf for a normal distribution
        '''
        C = - 0.5 * math.log(2 * math.pi)
        return C - self.sigma.log() - 0.5 * ((z - self.mu)/self.sigma)**2


class VariationalInference(torch.nn.Module):
    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta

    def reduce(self, x):
        return x.view(x.size(0), -1).sum(dim=1)

    def forward(self, model, x):

        outputs = model(x)

        #Unpacking outputs
        px, pz, qz, z, rot = [outputs[k] for k in ["px", "pz", "qz", "z", "rot"]]

        # evaluate log probabilities
        #log_px = self.reduce(rot_img(px.log_prob(x),rot))
        log_px = self.reduce(px.log_prob(x))
        log_pz = self.reduce(pz.log_prob(z))
        log_qz = self.reduce(qz.log_prob(z))

        #Computing elbo px.log_prob(x)
        kld = log_qz - log_pz
        elbo = log_px - kld
        beta_elbo = log_px - self.beta*kld
        loss = -elbo.mean()

        # prepare the output
        with torch.no_grad():
            diagnostics = {'elbo': elbo, 'log_px':log_px, 'kld': kld}
            
        return loss, diagnostics, outputs
