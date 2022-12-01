import torch
from invariant_ae import AE
from invariant_vae import VAE
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
emb_dim = 32
hidden_dim = 128


CUDA = torch.cuda.is_available()
BATCH_SIZE = 64
SEED =  42
EPOCHS = 500
ZDIMS = 32
HIDDEN_UNITS = 32


device = torch.device("cuda" if CUDA else "cpu")
print(device)
torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)



mnist_train = torchvision.datasets.MNIST('data', train = True, download=True, transform = ToTensor())
train_loader = torch.utils.data.DataLoader(mnist_train,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True
                                          )



net = VAE(emb_dim, hidden_dim).to(device)

test_output = net(next(iter(train_loader))[0][0].to("cuda"))
stop
