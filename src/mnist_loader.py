
import numpy as np

import torch
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

        return {'data': image, 'label': label}

def get_loaders(dataClass, data_path, version, batch_size=64, shuffle=True):

    train_data = dataClass('train', data_path=data_path, version=version)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)

    val_data = dataClass('val', data_path=data_path, version=version)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=shuffle)

    test_data = dataClass('test', data_path=data_path, version=version)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)

    return {'train': train_loader, 'val': val_loader, 'test': test_loader}


if __name__ == '__main__':

    # Extract dataloaders
    loaders = get_loaders(MNISTDataset, data_path='data', version='rotated')
