def train_model(activation_name):
    spline_activation = get_activation_function(activation_name)
    model = SimpleKANModel(layer_sizes=[64, 128], spline_activation=spline_activation)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
