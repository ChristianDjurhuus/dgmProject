import torch
import torch.nn.functional as F
from torch.nn import Module
from e2cnn import gspaces
from e2cnn import nn
from src.invariant_vae.utils import rot_img, get_rotation_matrix, get_batch_norm, get_non_linearity


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
        self.pool3 = nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)

        # convolution 7 --> out
        # the old output type is the input type to the next layer
        in_type = out_type
        out_scalar_fields = nn.FieldType(self.r2_act, out_dim * [self.r2_act.trivial_repr])
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
        #x = self.pool1(x)
        x = self.block3(x)
        x = self.block4(x)
        #x = self.pool2(x)
        x = self.block5(x)
        x = self.block6(x)
        #x = self.pool3(x)
        x = self.block7(x)

        #x = x.tensor.squeeze(-1).squeeze(-1)
        x = x.tensor.mean(dim=(2, 3))
        x_0, x_1 = x[:, :self.out_dim], x[:, self.out_dim:]
        mu, log_var = x_0[:,:self.out_dim//2], x_0[:,self.out_dim//2:]
        #print("mu, var shapes: ", mu.shape, log_var.shape)
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
        x = torch.sigmoid(x)
        return x


class VAE(Module):
    def __init__(self, emb_dim, hidden_dim):
        super().__init__()
        self.encoder = Encoder(out_dim=emb_dim*2)
        self.decoder = Decoder(input_size=emb_dim, hidden_size = hidden_dim)

    def forward(self, x, do_rot=True):
        mu, log_var, v = self.encoder(x)
        z = self.reparameterize(mu, log_var)

        rot = get_rotation_matrix(v)
        #print(v, rot)
        y = self.decoder(z)
        if do_rot:
            y = rot_img(y, rot)
        return y, rot, mu, log_var

    def reparameterize(self, mu, logvar, inference=False):
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        if inference:
            return mu
        
        return mu + (eps * std)
