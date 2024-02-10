import torch
# import numpy as np


# we will implement an efficient graph neural network using pytorch
# this will be used to reparameterize the x variables
# we want to use the cg_modes to define the graph structure
# First, we implement the graph convolutional (GCN) layer
# The GCN layer will use the cg_modes in its forward pass
# so that we never have to explicitly compute the adjacency matrix

class GCN_CG(torch.nn.Module):
    def __init__(self, in_features, out_features, cg, num_cg, bias=True, skip_proj=False, add_self_loops=True):
        """This is a class to implement the graph convolutional layer using PyTorch.
        The graph convolutional layer uses the cg_modes to define the graph structure.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            cg_modes (torch.Tensor): The matrix of cg_modes.
            bias (bool, optional): Whether to include a bias term. Defaults to True.
            skip_proj (bool, optional): Whether to skip the projection to the graph space. Defaults to False.
            add_self_loops (bool, optional): Whether to add self loops to the graph. Defaults to True.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # self.cg = cg  # the cg object
        # to avoid memory leak, we only keep the cg_modes and cg_eigenvalues
        self.num_cg = num_cg
        self.cg_modes = cg.cg_modes[:, :num_cg]
        self.cg_eigenvalues = cg.cg_eigenvalues[:num_cg]
        self.skip_proj = skip_proj
        self.add_self_loops = add_self_loops
        self.get_cg_params()
        self.weight = torch.nn.Parameter(torch.randn(in_features, out_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        # add parameter for self loops if needed
        if add_self_loops:
            self.self_loops = torch.nn.Parameter(torch.randn(in_features, out_features))
        else:
            self.register_parameter('self_loops', None)
        self.reset_parameters()
        
    def get_cg_params(self,):
        # rescale the cg_modes by the eigenvalues
        self.cg_modes_scaled = self.cg_modes * torch.sqrt(self.cg_eigenvalues)[None, :]
        # also, keep the transpose of the scaled cg_modes for efficient computation
        self.cg_modes_scaled_T = self.cg_modes_scaled.T
        
    def reset_parameters(self):
        # initialize the weights using xavier initialization
        torch.nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            # initialize the bias to zero
            torch.nn.init.zeros_(self.bias)
        if self.add_self_loops:
            # initialize the self loops to normal 
            torch.nn.init.normal_(self.self_loops)
            
    def forward(self, x):
        # compute the graph convolution
        # assume x is of shape (n, in_features)
        if not self.skip_proj:
            # we project the output features to the graph space
            # here, the graph adjacency matrix is given by A = cg_modes @ cg_modes.T
            # so we can use the cg_modes to define the graph convolution
            # for efficiency, we won't explicitly compute A
            # instead we perform convolution using the cg_modes_scaled in two steps as
            z = self.cg_modes_scaled_T @ x
            # # also use the cg eigenvalues to scale the output
            # z = z * self.cg_eigenvalues[:, None]
            # then, we project the graph space back to the output features
            x1 = self.cg_modes_scaled @ z
            # # add self loops
            # if self.add_self_loops:
            #     x = x1 + x
            # else:
            #     x = x1
        else:
            x1 = x   
        # we project the input features to the output features
        x1 = x1 @ self.weight
        # add self loops
        if self.add_self_loops:
            x1 = x1 + x @ self.self_loops
        # add bias
        if self.bias is not None:
            x1 = x1 + self.bias
        return x1

# Define a resisual block using the GCN layer
# The residual block will use the GCN layer to perform the graph convolution

class ResGCN_CG(torch.nn.Module):
    def __init__(self, in_features, out_features, cg,num_cg, bias=True, activation=torch.nn.ReLU()):
        """This is a class to implement the residual block using the graph convolutional layer.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            cg_modes (torch.Tensor): The matrix of cg_modes.
            bias (bool, optional): Whether to include a bias term. Defaults to True.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_cg = num_cg
        self.gcn1 = GCN_CG(in_features, out_features, cg, num_cg, bias)
        self.gcn2 = GCN_CG(out_features, out_features, cg, num_cg, bias)
        self.act = activation
        # because the input and output dimensions are different, we need to project the input to the output
        self.proj = torch.nn.Linear(in_features, out_features)
        
    def forward(self, x):
        # compute the residual block
        # assume x is of shape (n, in_features)
        # first, we perform the graph convolution using the first GCN layer
        x1 = self.gcn1(x)
        # then, we apply the ReLU activation
        x1 = self.act(x1)
        # then, we perform the graph convolution using the second GCN layer
        x2 = self.gcn2(x1)
        # finally, we add the input to the output
        # note that we cannot add the input to the output if the input 
        # and output dimensions are different
        x = x2 + self.proj(x)
        return x

# Next, we implement the graph neural network (GNN) using the GCN layer
# the GNN will use the GCN layer to perform the graph convolution
# it will take a set of hidden dimensions, including the input and output dimensions


class GNNRes(torch.nn.Module):
    def __init__(self, hidden_dims, cg, num_cg , bias=True, activation=torch.nn.ReLU()):
        """This is a class to implement the graph neural network using the residual GCN block.
        It will also have a final linear layer to project the output to the desired dimension.

        Args:
            hidden_dims (list): The list of hidden dimensions. 
                The first element is the input dimension and 
                the last element is the output dimension. 
                It should have at least two elements.
            cg (object): The object containing the cg_modes and cg_eigenvalues.
            num_cg (int): The number of cg_modes to use.
            bias (bool, optional): Whether to include a bias term. 
                Defaults to True.
            activation (torch.nn.Module, optional): The activation function. 
                Defaults to torch.nn.ReLU().
        """
        super().__init__()
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims) - 1
        self.num_cg = num_cg
        self.layers = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.layers.append(ResGCN_CG(hidden_dims[i], hidden_dims[i+1], cg, num_cg, bias, activation))
        # the final layer is a linear layer
        self.layers.append(torch.nn.Linear(hidden_dims[-2], hidden_dims[-1]))
        self.act = activation
        
    def forward(self, x):
        # compute the graph neural network
        # assume x is of shape (n, in_features)
        for i in range(self.num_layers - 1):
            x = self.layers[i](x)
            # apply the activation
            x = self.act(x)
        # apply the final layer
        x = self.layers[-1](x)
        return x        
    
# Next, we implement the graph neural network (GNN) using the GCN layer

class GNN(torch.nn.Module):
    def __init__(self, hidden_dims, cg, num_cg, bias=True, activation=torch.nn.ReLU()):
        """This is a class to implement the graph neural network using the GCN layer.
        It will also have a final linear layer to project the output to the desired dimension.

        Args:
            hidden_dims (list): The list of hidden dimensions. 
                The first element is the input dimension and 
                the last element is the output dimension. 
                It should have at least two elements.
            cg (object): The object containing the cg_modes and cg_eigenvalues.
            num_cg (int): The number of cg_modes to use.
            bias (bool, optional): Whether to include a bias term. 
                Defaults to True.
            activation (torch.nn.Module, optional): The activation function. 
                Defaults to torch.nn.ReLU().
        """
        super().__init__()
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims) - 1
        self.num_cg = num_cg
        self.layers = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.layers.append(GCN_CG(hidden_dims[i], hidden_dims[i+1], cg, num_cg, bias))
        # the final layer is a linear layer
        self.layers.append(torch.nn.Linear(hidden_dims[-2], hidden_dims[-1]))
        self.act = activation
        
    def forward(self, x):
        # compute the graph neural network
        # assume x is of shape (n, in_features)
        for i in range(self.num_layers - 1):
            x = self.layers[i](x)
            # apply the activation
            x = self.act(x)
        # apply the final layer
        x = self.layers[-1](x)
        return x

# GNN reparemeterization:
# we will use the GNN to reparameterize the x variables
# we introduce a convenience class to perform the reparameterization
# this class includes both the GNN and the initial position as parameters
# it will take the same inputs as the GNN class and return the reparameterized x

class GNNReparam(torch.nn.Module):
    def __init__(self, hidden_dims, cg, num_cg, latent_sigma, bias=True, activation=torch.nn.ReLU(), GNN_class=GNN):
        """This is a class to implement the graph neural network reparameterization.
        

        Args:
            hidden_dims (list): The list of hidden dimensions. 
                The first element is the input dimension and 
                the last element is the output dimension. 
                It should have at least two elements.
            cg (object): The object containing the cg_modes and cg_eigenvalues.
            num_cg (int): The number of cg_modes to use.
            initial_pos (torch.Tensor): The initial position of the particles.
            bias (bool, optional): Whether to include a bias term. 
                Defaults to True.
            activation (torch.nn.Module, optional): The activation function. 
                Defaults to torch.nn.ReLU().
        """
        super().__init__()
        self.hidden_dims = hidden_dims
        self.num_cg = num_cg
        self.gnn = GNN_class(hidden_dims, cg, num_cg, bias, activation)
        self.n = cg.cg_modes.shape[0]
        self.get_latent_embedding(latent_sigma)
        
    def get_latent_embedding(self, latent_sigma):
        # get the latent embedding
        # we use the latent_sigma to scale the initial position
        self.latent_embedding = torch.nn.Parameter(latent_sigma * torch.randn(self.n, self.gnn.hidden_dims[0]))
                
    def forward(self):
        # compute the reparameterized x
        # assume x is of shape (n, in_features)
        return self.gnn(self.latent_embedding)