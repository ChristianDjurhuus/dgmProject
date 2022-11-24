
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm

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


def get_loaders(dataClass, data_path, version, batch_size=64):

    train_data = dataClass('train', data_path=data_path, version=version)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    val_data = dataClass('val', data_path=data_path, version=version)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

    test_data = dataClass('test', data_path=data_path, version=version)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return {'train': train_loader, 'val': val_loader, 'test': test_loader}


"""
def load_mnist(data_path, batch_size=None, mnist_type='regular'):
    data = {'train': defaultdict(dict), 'test': defaultdict(dict)}

    # Load MNIST train split
    if mnist_type == 'rotated':
        train_data = np.loadtxt(f'{data_path}/mnist_rotated/mnist_all_rotation_normalized_float_train_valid.amat')
        test_data = np.loadtxt(f'{data_path}/mnist_rotated/mnist_all_rotation_normalized_float_test.amat')
    else:
        train_data = np.loadtxt(f'{data_path}/mnist/mnist_train.amat')
        test_data = np.loadtxt(f'{data_path}/mnist/mnist_test.amat')

    # Batch or complete data
    if batch_size == None:
        data['train']['data'] = torch.tensor(train_data[:, :-1]).to(torch.float32)
        data['train']['labels'] = torch.tensor(train_data[:, -1]).to(torch.float32)
        data['test']['data'] = torch.tensor(test_data[:, :-1]).to(torch.float32)
        data['test']['labels'] = torch.tensor(test_data[:, -1]).to(torch.float32)

    else: # Batch it up!
        for batch in tqdm(range(train_data.shape[0] // batch_size)):
            batch_start, batch_end = batch*batch_size, (batch+1)*batch_size
            data['train'][batch]['data'] = torch.tensor(train_data[:, :-1][batch_start:batch_end]).to(torch.float32).round()
            data['train'][batch]['labels'] = torch.tensor(train_data[:, -1][batch_start:batch_end]).to(torch.float32)

        # Create a batch for the remaining data points
        if train_data.shape[0] % batch_size != 0:
            batch += 1
            batch_start, batch_end = batch*batch_size, train_data.shape[0]
            data['train'][batch]['data'] = torch.tensor(train_data[:, :-1][batch_start:batch_end]).to(torch.float32).round()
            data['train'][batch]['labels'] = torch.tensor(train_data[:, -1][batch_start:batch_end]).to(torch.float32)

        # Load MNIST test split
        for batch in tqdm(range(test_data.shape[0] // batch_size)):
            batch_start, batch_end = batch*batch_size, (batch+1)*batch_size
            data['test'][batch]['data'] = torch.tensor(test_data[:, :-1][batch_start:batch_end]).to(torch.float32).round()
            data['test'][batch]['labels'] = torch.tensor(test_data[:, -1][batch_start:batch_end]).to(torch.float32)

        # Create a batch for the remaining data points
        if test_data.shape[0] % batch_size != 0:
            batch += 1
            batch_start, batch_end = batch*batch_size, test_data.shape[0]
            data['test'][batch]['data'] = torch.tensor(test_data[:, :-1][batch_start:batch_end]).to(torch.float32).round()
            data['test'][batch]['labels'] = torch.tensor(test_data[:, -1][batch_start:batch_end]).to(torch.float32)

    return data
"""

if __name__ == '__main__':

    # Extract dataloaders
    loaders = get_loaders(MNISTDataset, data_path='data', version='rotated')
