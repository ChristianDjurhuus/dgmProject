
import torch
import numpy as np
from collections import defaultdict

from tqdm import tqdm

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