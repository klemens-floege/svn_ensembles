import torch


def initliase_mlp_models(input_dim, output_dim, config):
    # Dynamically adjust layer sizes
    n_particles = config.experiment.n_particles
    hidden_layers = config.experiment.hidden_layers


    layer_sizes = [input_dim]  # Set the first layer size to match the feature dimension of x_train
    layer_sizes.extend(hidden_layers)
    layer_sizes.append(output_dim)  # Set the last layer size to match the output dimension

    print('Layer sizes:', layer_sizes)

    modellist = []

    for _ in range(n_particles):
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            # Add ReLU activation, except for the output layer
            if i < len(layer_sizes) - 2:
                layers.append(torch.nn.ReLU())
            
        if config.task.task_type == 'classification':
        # For classification, typically use a softmax for multi-class or sigmoid for binary classification
            if output_dim == 1:
                layers.append(torch.nn.Sigmoid())  # Binary classification
            else:
                layers.append(torch.nn.Softmax(dim=1))  # Multi-class classification
                
                #self.layer2 = nn.Linear(64, n_classes)  # output layer with a neuron for each class
                #self.softmax = nn.Softmax(dim=1)  # Softmax over the second dimension
            

        #TODO: for classifiation include output activation as well
        model = torch.nn.Sequential(*layers)
        modellist.append(model)

    return modellist