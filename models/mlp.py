import torch.nn as nn
import torch
import numpy as np
import math
import torch.nn.functional as F

def dnorm2( X, Y):
    XX = X.matmul(X.t())
    XY = X.matmul(Y.t())
    YY = Y.matmul(Y.t())
    dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)
    return dnorm2 


class Net(nn.Module):
    """
    Implementation of a fully connected neural network using an nn.ModuleList
    for compatibility with BackPACK, with support for custom weights in forward pass.
    
    Args:
        layer_sizes (list): List containing the sizes of each layer.
        classification (bool): If the network is for a classification task.
        act (callable): Activation function for hidden layers.
        out_act (callable): Activation function for the output layer. If None, linear is used.
        bias (bool): Whether to include biases in the layers.
    """
    
    def __init__(self, layer_sizes, classification=False, act=F.relu, out_act=None, bias=True,  no_weights = True):
        super(Net, self).__init__()
        self.layer_sizes = layer_sizes
        self.layers = nn.ModuleList()
        self.classification = classification
        self.act = act
        self.out_act = out_act
        self.bias = bias

        
        
        # Create layers
        for in_features, out_features in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.layers.append(nn.Linear(in_features, out_features, bias=bias))

        self.num_params = sum(p.numel() for p in self.parameters())

        self.param_shapes = [list(i.shape) for i in self.parameters()]

        self._weights = None
        if no_weights:
            return
        
        ### Define and initialize network weights.
        # Each odd entry of this list will contain a weight Tensor and each
        # even entry a bias vector.
        self._weights = nn.ParameterList()

        for i, dims in enumerate(self.param_shapes):
            self._weights.append(nn.Parameter(torch.Tensor(*dims),
                                              requires_grad=True))


        for i in range(0, len(self._weights), 2 if bias else 1):
            if bias:
                self.init_params(self._weights[i], self._weights[i + 1])
            else:
                self.init_params(self._weights[i])

    def init_params(self,weights, bias=None):
        """Initialize the weights and biases of a linear or (transpose) conv layer.

        Note, the implementation is based on the method "reset_parameters()",
        that defines the original PyTorch initialization for a linear or
        convolutional layer, resp. The implementations can be found here:

            https://git.io/fhnxV
        Args:
            weights: The weight tensor to be initialized.
            bias (optional): The bias tensor to be initialized.
        """
        nn.init.kaiming_uniform_(weights, a=math.sqrt(5))
        if bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(bias, -bound, bound)
            
    def forward(self, x, weights=None):
        """
        Forward pass through the network with optional custom weights.
        
        Args:
            x (torch.Tensor): Input tensor.
            weights (list of torch.Tensor, optional): Custom weights for the network. 
                Each element of the list corresponds to the weights and biases (if bias=True)
                of each layer, in order. If None, the model's current weights are used.
            
        Returns:
            torch.Tensor: Output of the network.
        """
        if weights is None:
            # Use model's current weights if no custom weights provided
            for layer in self.layers[:-1]:
                x = layer(x)
                if self.act is not None:
                    x = self.act(x)
            x = self.layers[-1](x)  # Apply last layer without activation
        else:
            # Use custom weights provided
            for i, layer in enumerate(self.layers[:-1]):
                weight, bias = weights[2*i], weights[2*i+1]
                x = F.linear(x, weight, bias)
                if self.act is not None:
                    x = self.act(x)
            # Apply weights for the last layer
            weight, bias = weights[-2], weights[-1]
            x = F.linear(x, weight, bias)
        
        if self.out_act is not None:
            x = self.out_act(x)
            
        return x

'''class Net(torch.nn.Module):
    """
    Implementation of Fully connected neural network

    Args:
        layer_sizes(list): list containing the layer sizes
        classification(bool): if the net is used for a classification task
        act: activation function in the hidden layers
        out_act: activation function in the output layer, if None then linear
        bias(Bool): whether or not the net has biases
    """

    def __init__(self, layer_sizes, classification = False, act=F.sigmoid,d_logits = False, out_act=None, bias=True, no_weights = True):
        super(Net, self).__init__()
        self.layer_sizes = layer_sizes
        self.layer_list = []
        self.classification = classification
        self.bias = bias
        self.d_logits = d_logits
        self.ac = act
        self.out_act = out_act
        for l in range(len(layer_sizes[:-1])):
            layer_l = nn.Linear(layer_sizes[l], layer_sizes[l+1], bias=self.bias)
            self.add_module('layer_' + str(l), layer_l)

        self.num_params = sum(p.numel() for p in self.parameters())

        self.param_shapes = [list(i.shape) for i in self.parameters()]

        self._weights = None
        if no_weights:
            return

        ### Define and initialize network weights.
        # Each odd entry of this list will contain a weight Tensor and each
        # even entry a bias vector.
        self._weights = nn.ParameterList()

        for i, dims in enumerate(self.param_shapes):
            self._weights.append(nn.Parameter(torch.Tensor(*dims),
                                              requires_grad=True))


        for i in range(0, len(self._weights), 2 if bias else 1):
            if bias:
                self.init_params(self._weights[i], self._weights[i + 1])
            else:
                self.init_params(self._weights[i])


    def init_params(self,weights, bias=None):
        """Initialize the weights and biases of a linear or (transpose) conv layer.

        Note, the implementation is based on the method "reset_parameters()",
        that defines the original PyTorch initialization for a linear or
        convolutional layer, resp. The implementations can be found here:

            https://git.io/fhnxV
        Args:
            weights: The weight tensor to be initialized.
            bias (optional): The bias tensor to be initialized.
        """
        nn.init.kaiming_uniform_(weights, a=math.sqrt(5))
        if bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(bias, -bound, bound)


    def forward(self, x, weights=None, ret_pre = False):
        """Can be used to make the forward step and make predictions.

        Args:
            x(torch tensor): The input batch to feed the network.
            weights(list): A reshaped particle
        Returns:
            (tuple): Tuple containing:

            - **y**: The output of the network
            - **hidden** (optional): if out_act is not None also the linear output before activation is returned
        """
        
        #print('before if statement', weights)

        if weights is None:
            weights = self._weights
        else:
            shapes = self.param_shapes
            #print('length shapes', len(shapes))
            #print('len weights', len(weights))
            assert (len(weights) == len(shapes))
            for i, s in enumerate(shapes):
                assert (np.all(np.equal(s, list(weights[i].shape))))
        
        #print('before after statement', weights)

        #print('selfwe weights', self._weights)

        hidden = x

        if self.bias:
            num_layers = len(weights) // 2
            step_size = 2
        else:
            num_layers = len(weights)
            step_size = 1

        for l in range(0, len(weights), step_size):
            W = weights[l]
            if self.bias:
                b = weights[l + 1]
            else:
                b = None

            if l==len(weights)-2 and self.d_logits:
                pre_out = hidden
                distance_logits = dnorm2(pre_out, W)

            hidden = F.linear(hidden, W, bias=b)

            # Only for hidden layers.
            if l / step_size + 1 < num_layers:
                if self.ac is not None:
                    hidden = self.ac(hidden)

        if self.d_logits:
            hidden = -distance_logits
        if self.out_act is not None:
            return self.out_act(hidden), hidden #needed so that i can use second output for training first for predict
        else:
            return hidden'''