import pickle
import numpy as np
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from pykan import KANConv2DLayer, BSplineActivation, PolynomialActivation, RBFActivation, WaveletActivation


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

# Define the KAN model
class SimpleKANModel(nn.Module):
    def __init__(self, layer_sizes, spline_activation):
        super(SimpleKANModel, self).__init__()

        self.layers = nn.Sequential(
            KANConv2DLayer(in_channels=3, out_channels=layer_sizes[0], activation=spline_activation),
            nn.ReLU(),
            KANConv2DLayer(in_channels=layer_sizes[0], out_channels=layer_sizes[1], activation=spline_activation),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(layer_sizes[1], 10)
        self.drop = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = self.fc(x)
        return x

# Function to get activation function
def get_activation_function(name):
    if name == 'b_spline':
        return BSplineActivation()
    if name == 'polynomial':
        return PolynomialActivation()
    if name == 'rbf':
        return RBFActivation()
    if name == 'wavelet':
        return WaveletActivation()
    else:
        raise ValueError("Unsupported activation function")

# Load the CIFAR-10 data
data_dir = './cifar-10-batches-py'
(train_data, train_labels), (test_data, test_labels) = load_cifar10_data(data_dir)

# Prepare the data for PyTorch
train_loader, test_loader = prepare_data_for_pytorch(train_data, train_labels, test_data, test_labels)

# Define hyperparameters
layer_sizes = [32, 64]
activation_name = 'b_spline'
spline_activation = get_activation_function(activation_name)

# Instantiate the model
model = SimpleKANModel(layer_sizes, spline_activation)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

print("Training complete")
