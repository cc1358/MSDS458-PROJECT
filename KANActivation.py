import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

# Create Class for KANConv2DLayer
class KANConv2DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None):
        super(KANConv2DLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

# Create Class for B-Spline
class BSplineActivation(nn.Module):
    def __init__(self, degree=3):
        super(BSplineActivation, self).__init__()
        self.degree = degree

    def forward(self, x):
      
        y = torch.zeros_like(x)
        for i in range(1, self.degree + 1):
            y += torch.pow(x, i)
        return y

# Create Cubic Spline Class
class CubicSplineActivation(nn.Module):
    def __init__(self):
        super(CubicSplineActivation, self).__init__()
        # Initialize coefficients
        self.coefficients = nn.Parameter(torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32))
        # Clamp range
        self.clamp_min = -1.0
        self.clamp_max = 1.0

    def forward(self, x):
        # Clamp the input to the range to simulate spline behavior
        x_clamped = torch.clamp(x, min=self.clamp_min, max=self.clamp_max)

        # Compute the cubic polynomial activation function
        y = (self.coefficients[0] +
             self.coefficients[1] * x_clamped +
             self.coefficients[2] * torch.pow(x_clamped, 2) +
             self.coefficients[3] * torch.pow(x_clamped, 3))
        return y

# Create Swish Activation Class
class SwishActivation(nn.Module):
    def __init__(self, beta=1.0):
        super(SwishActivation, self).__init__()
        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

# Create Wavelet Activation Class
class WaveletActivation(nn.Module):
    def __init__(self, wavelet_type='haar'):
        super(WaveletActivation, self).__init__()
        self.wavelet_type = wavelet_type

    def forward(self, x):
        if self.wavelet_type == 'haar':
            return torch.sign(x) * (1.0 - torch.abs(x))
        elif self.wavelet_type == 'mexican_hat':
            return (1.0 - torch.pow(x, 2)) * torch.exp(-0.5 * torch.pow(x, 2))
        else:
            raise ValueError(f"Unsupported wavelet type: {self.wavelet_type}")

# Create KAN Model Class
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

# Retrieve Activation Function
def get_activation_function(name):
    if name == 'b_spline':
        return BSplineActivation(degree=3)
    if name == 'cubic':
        return CubicSplineActivation()
    if name == 'swish':
        return SwishActivation()
    if name == 'wavelet':
        return WaveletActivation()
    else:
        raise ValueError("Unsupported activation function")

# Training and Evaluation Functions
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    start_time = time.time()
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
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Training completed in: {runtime:.2f} seconds")
    return runtime

def evaluate_model(model, test_loader):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    print(f'Test Precision: {precision * 100:.2f}%')
    print(f'Test Recall: {recall * 100:.2f}%')
    print(f'Test F1 Score: {f1 * 100:.2f}%')

    return accuracy, precision, recall, f1

# Load CIFAR-10 and Test and Train
if __name__ == "__main__":
    # Load the CIFAR-10 data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Define hyperparameters
    layer_sizes = [32, 64]
    num_epochs = 10

    activation_functions = ['b_spline', 'cubic', 'swish', 'wavelet']
    results = {}

    for activation_name in activation_functions:
        print(f'\nRunning model with {activation_name} activation function...')
        spline_activation = get_activation_function(activation_name)
        model = SimpleKANModel(layer_sizes, spline_activation)

        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        runtime = train_model(model, train_loader, criterion, optimizer, num_epochs)

        # Evaluate the model
        accuracy, precision, recall, f1 = evaluate_model(model, test_loader)

        # Store the results
        results[activation_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'runtime': runtime
        }

    # Print out the results
    print("\nFinal Results:")
    for activation_name, metrics in results.items():
        print(f"{activation_name} Activation - Test Accuracy: {metrics['accuracy'] * 100:.2f}%, "
              f"Precision: {metrics['precision'] * 100:.2f}%, "
              f"Recall: {metrics['recall'] * 100:.2f}%, "
              f"F1 Score: {metrics['f1'] * 100:.2f}%, "
              f"Runtime: {metrics['runtime']:.2f} seconds")
