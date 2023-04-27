import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from typing import Callable

class DGCF(nn.Module):
    r"""Performs dynamic spatial convolution on graphs using a fixed-sized kernel.

    params:
    - n_nodes, int: number of nodes of the input graphs
    - kernel_size, int: number of neighbors for convolution
    - neighborhoods, torch.Tensor: matrix with dimensions (n_nodes, kernel_size) where the entry [Q]_ij 
            denotes for the i's j-th closest neighbor.
    - in_channels, int: number of input channels for each node
    - out_channels, int: number of output channels for each node
    - filter_generating_network, nn.Module: filter generating network. It receives in input a feature vector resulting by
            the vectorization of all the nodes' features. It returns the vectorized weights 
            (whose dimension is out_channels * in_channels * kernel_size [+ out_channels if bias is dynamic]).
    - bias, string (default, 'static'):, whether the layer uses a bias vector. It can be:
            - 'static', whether the layer uses a single bias vector for each input
            - 'dynamic', whether the biases are generated from the filter_generating_network. In such case, the biases
              will be determined by the last [+ out_channels] weights generated from the filter_generating network.
    - bias_initializer, Callable: Initializer for the bias vector (zeros by default. Use nn.init.* inizializations)
            
    # Input shape    
        3D tensor with shape:
        (batch_size, n_nodes, in_channels).
    # Output shape
        3D tensor with shape:
        (batch_size, n_nodes, out_channels).

    Examples::
        >>> ...
        >>> # Supposing to have a batch of 20 graphs with n_nodes=5 and in_channels=3 per node
        >>> ...
        >>> print(x.size())
        torch.Size([20, 5, 3])
        >>> out_features = 1
        >>> # Let's define the dynamic-filter network's architecture
        >>> filter_generating_net = nn.Sequential(
        >>> ... nn.Linear(in_channels * n_nodes, 50),
        >>> ... nn.ReLU(),
        >>> ... nn.Linear(50, out_channels * in_channels * kernel_size)
        >>> )
        >>> f = DGCF(n_nodes, kernel_size, neighborhoods, in_channels, out_channels, filter_generating_net)
        >>> y = f(x)
        >>> print(y.size())
        torch.Size([20, 5, 1])
    """

    def __init__(self, n_nodes: int, 
                       kernel_size: int, 
                       neighborhoods: torch.Tensor, 
                       in_channels: int, 
                       out_channels: int,
                       filter_generating_network: nn.Sequential,
                       bias: str = 'static',
                       bias_initializer: Callable = nn.init.zeros_) -> None:
        super(DGCF, self).__init__()
        # Data storage
        self.n_nodes = n_nodes
        self.kernel_size = kernel_size
        self.neighborhoods = neighborhoods
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_generating_network = filter_generating_network
        self.use_bias = bias
        self.register_buffer("zero_node", torch.zeros(1, in_channels))

        if self.use_bias == 'static':
            self.bias = Parameter(bias_initializer(torch.empty(1, 1, out_channels)))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        # Computing dynamic biases
        dynamic_weights = self.filter_generating_network(batch.flatten(1))
        # Extracting biases and filters
        if self.use_bias == 'dynamic':
            bias = dynamic_weights[:,-self.out_channels:]
            bias = bias.unsqueeze(1)
            dynamic_weights = dynamic_weights[:,:-self.out_channels]
        elif self.use_bias == 'static':
            bias = self.bias
        # Reshaping kernels
        dynamic_weights = dynamic_weights.reshape(-1, self.kernel_size, self.in_channels, self.out_channels)
        # Appending padding node
        batch = torch.cat((batch, self.zero_node.repeat(len(batch), 1, 1)), 1)
        # Extracting neighborhoods
        batch = batch[:, self.neighborhoods, :]
        # Applying dynamic convolution
        output = []
        for i, kernel in enumerate(dynamic_weights):
            output.append(torch.tensordot(batch[i], kernel, dims=([1,2],[0,1])))
        output = torch.stack(output)
        # Summing biases
        if self.use_bias != 'none':
            output += bias
        return output

    def extra_repr(self) -> str:
        return 'in_channels={}, out_channels={}, bias={}, kernel_size={}\nFGN={}'.format(
            self.in_channels, self.out_channels, self.use_bias, self.kernel_size, self.filter_generating_network
        )
