import pickle
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10_data(data_dir):
    train_data = []
    train_labels = []
    for i in range(1, 6):
        data_dict = unpickle(os.path.join(data_dir, f'data_batch_{i}'))
        train_data.append(data_dict[b'data'])
        train_labels.append(data_dict[b'labels'])
    train_data = np.concatenate(train_data)
    train_labels = np.concatenate(train_labels)

    test_data_dict = unpickle(os.path.join(data_dir, 'test_batch'))
    test_data = test_data_dict[b'data']
    test_labels = np.array(test_data_dict[b'labels'])

    return (train_data, train_labels), (test_data, test_labels)

def prepare_data_for_pytorch(train_data, train_labels, test_data, test_labels):
    # Normalize the data
    train_data = train_data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    test_data = test_data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0

    # Convert to PyTorch tensors
    train_data = torch.tensor(train_data)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_data = torch.tensor(test_data)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    # Create datasets and dataloaders
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader

