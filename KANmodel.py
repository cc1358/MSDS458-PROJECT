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
