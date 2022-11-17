import torch
import torch.nn.functional as F
from torch.nn import Module
from e2cnn import gspaces
from e2cnn import nn

def get_rotation_matrix(v, eps=10e-5):
    v = v / (torch.norm(v, dim=-1, keepdim=True) + eps)
    rot = torch.stack((
        torch.stack((v[:, 0], v[:, 1]), dim=-1),
        torch.stack((-v[:, 1], v[:, 0]), dim=-1),
        torch.zeros(v.size(0), 2).type_as(v)
    ), dim=-1)
    return rot

def rot_img(x, rot):
    grid = F.affine_grid(rot, x.size(), align_corners=False).type_as(x)
    x = F.grid_sample(x, grid, align_corners=False)
    return x

def get_non_linearity(scalar_fields, vector_fields):
    out_type = scalar_fields + vector_fields
    relu = nn.ReLU(scalar_fields)
    norm_relu = nn.NormNonLinearity(vector_fields)
    nonlinearity = nn.MultipleModule(
        out_type,
        ['relu'] * len(scalar_fields) + ['norm'] * len(vector_fields),
        [(relu, 'relu'), (norm_relu, 'norm')]
    )
    return nonlinearity

def get_batch_norm(scalar_fields, vector_fields):
    out_type = scalar_fields + vector_fields
    batch_norm = nn.InnerBatchNorm(scalar_fields)
    norm_batch_norm = nn.NormBatchNorm(vector_fields)
    batch_norm = nn.MultipleModule(
        out_type,
        ['bn'] * len(scalar_fields) + ['nbn'] * len(vector_fields),
        [(batch_norm, 'bn'), (norm_batch_norm, 'nbn')]
    )
    return batch_norm

