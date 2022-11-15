
import torch
import numpy as np

def load_mnist(device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
    data = {'train': {}, 'test': {}}

    # Load MNIST train split
    train_data = np.loadtxt('data/mnist/mnist_train.amat')
    data['train']['data'] = torch.tensor(train_data[:, :-1]).to(device)
    data['train']['labels'] = torch.tensor(train_data[:, -1]).to(device)

    # Load MNIST test split
    test_data = np.loadtxt('data/mnist/mnist_test.amat')
    data['train']['data'] = torch.tensor(test_data[:, :-1]).to(device)
    data['train']['labels'] = torch.tensor(test_data[:, -1]).to(device)
    return data

def load_rotated_mnist(device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
    data = {'train': {}, 'test': {}}

    # Load MNIST train split
    train_data = np.loadtxt('data/mnist_rotated/mnist_all_rotation_normalized_float_train_valid.amat')
    data['train']['data'] = torch.tensor(train_data[:, :-1]).to(device)
    data['train']['labels'] = torch.tensor(train_data[:, -1]).to(device)

    # Load MNIST test split
    test_data = np.loadtxt('data/mnist_rotated/mnist_all_rotation_normalized_float_test.amat')
    data['train']['data'] = torch.tensor(test_data[:, :-1]).to(device)
    data['train']['labels'] = torch.tensor(test_data[:, -1]).to(device)
    return data
