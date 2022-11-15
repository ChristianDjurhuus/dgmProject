import torch.nn 
import torch
import torch.nn.functional as F
import e2cnn



class Encoder():
    '''
    ENCODER:
    7 layers steerable CNNs
    No pooling (breaks rotational invariance)

    Average over the two spatial dimensions after the final layer to extract final invariant embedding and equivariant vector
    
    each layer have 32 hidden scalar and 32 hidden vector fields
    final layer -> 32 scalar fields + one vector feature field


    Sources:
    https://quva-lab.github.io/e2cnn/api/e2cnn.group.html#e2cnn.group.Representation
    https://quva-lab.github.io/e2cnn/api/e2cnn.gspaces.html#e2cnn.gspaces.Rot2dOnR2
    https://github.com/QUVA-Lab/e2cnn/blob/master/examples/model.ipynb
    
    '''
    def __init__(self, hidden_units, latent_dim):
        super(Encoder, self).__init__()
        self.hidden_units = hidden_units
        self.latent_dim = latent_dim




    def forward(self):


        return None

class Decoder():
    '''
    DECODER:
    Regular CNN
    six layers of regular CNNs with 32 hidden channels
    interleaved bilinear upsampling layers starting from the embedding expanded to a 2x2x32 tensor
    
    TODO: Maybe add dropout '''

    def __init__(self, input_size, hidden_units):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_units = hidden_units


    def forward(self):

        return None

class VAE():
    def __init__(self):
        self. = 


    def forward(self):

        return None
