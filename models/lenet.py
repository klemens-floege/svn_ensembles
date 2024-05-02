import torch
import torch.nn as nn



def initliase_lenet_models(input_dim, output_dim, config):

    n_particles = config.experiment.n_particles

    modellist = []

    for _ in range(n_particles):
        
        #TODO: Double check LeNet Architecture
        lenet_sequential = nn.Sequential(
            #nn.Unflatten(1, (1, input_dim, input_dim)),  # Unflatten the input into size (1, 28, 28)
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),  # Same padding to keep dimensions
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),  # Flatten the output for the fully connected layer
            nn.Linear(16 * 5 * 5, 120),  # First fully connected layer
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, output_dim)  
        )
        
        modellist.append(lenet_sequential)

    return modellist