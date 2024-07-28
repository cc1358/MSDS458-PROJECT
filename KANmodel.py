import torch.nn as nn
from pykan import KANConv2DLayer

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
