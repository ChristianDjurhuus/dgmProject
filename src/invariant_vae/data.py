import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms
from pytorch_lightning import LightningDataModule
from torch.utils.data import Subset
from torch.utils.data import Dataset


class MNISTDataset(Dataset):

    def __init__(self, setname, data_path, version: str = 'original'):
        assert setname in ['train', 'val', 'test']
        assert version in ['original', 'rotated']

        self.setname = setname
        self.version = version

        if version == 'original':
            filename = "mnist_test.amat" if setname == 'test' else "mnist_train.amat"
        else:
            filename = "mnist_all_rotation_normalized_float_test.amat" if setname == 'test' else "mnist_all_rotation_normalized_float_train_valid.amat"

        data = np.loadtxt(f"{data_path}/mnist_{version}/{filename}")
        data = data[:int(len(data) * 0.8), :] if setname == 'train' else data[:int(len(data) * 0.2), :] if setname == 'val' else data
        self.dataset = torch.Tensor(data[:, :-1]).to(torch.float32).bernoulli()
        self.labels = torch.Tensor(data[:, -1]).to(torch.float32)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image = self.dataset[item, :]
        label = self.labels[item]

        return image,label#{'data': image, 'label': label}



class MNISTDataModule(LightningDataModule):
    def __init__(self, data_dir: str = "data", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
               # transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.dims = (1, 28, 28)
        self.num_classes = 10

    def prepare_data(self):
        #torch builtin mnist
        #torchvision.datasets.MNIST(self.data_dir, train=True, download=True)
        #torchvision.datasets.MNIST(self.data_dir, train=False, download=True)

        #dataset from paper
        pass

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            #builtin VVV
            #mnist_full = torchvision.datasets.MNIST(self.data_dir, train=True, transform=self.transform)
            ####
            #subset_mnist = Subset(mnist_full, indices=range(len(mnist_full) // 10))
            #self.mnist_train, self.mnist_val = random_split(subset_mnist, [5000, 1000] )
            ####
            #self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
            
            #paper dataset:
            self.mnist_train = MNISTDataset("train", self.data_dir)
            self.mnist_val = MNISTDataset("val", self.data_dir)


        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            #self.mnist_test = torchvision.datasets.MNIST(self.data_dir, train=False, transform=self.transform)
            self.mnist_test = MNISTDataset("test", self.data_dir)
    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)