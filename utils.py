from scipy.sparse import coo_matrix
import torch
import networkx as nx
import torch.nn as nn

class ConvGNN(torch.nn.Module):
    r"""PyTorch adaptation of 'A Generalization of Convolutional Neural Networks to Graph-Structured Data' - Hechtlinger et al. 
    Keras original source: https://github.com/hechtlinger/graph_cnn
    """

    def __init__(self, n_nodes: int, 
                       kernel_size: int, 
                       neighborhoods: torch.Tensor, 
                       in_channels: int, 
                       out_channels: int, 
                       bias: bool = True,
                       kernel_initializer = torch.nn.init.kaiming_uniform_,
                       bias_initializer = torch.nn.init.zeros_) -> None:
        super(ConvGNN, self).__init__()
        # Data storage
        self.n_nodes = n_nodes
        self.kernel_size = kernel_size
        self.neighborhoods = neighborhoods
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = bias
        self.register_buffer("zero_node", torch.zeros(1, self.in_channels))

        self.kernel = nn.Parameter(kernel_initializer(torch.empty(kernel_size, in_channels, out_channels)))
        if self.use_bias:
            self.bias = nn.Parameter(bias_initializer(torch.empty(1, 1, out_channels)))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        # Appending padding node
        batch = torch.cat((batch, self.zero_node.repeat(len(batch), 1, 1)), 1)
        # Extracting neighborhoods
        batch = batch[:, self.neighborhoods, :]
        # Applying convolution
        output = torch.tensordot(batch, self.kernel, dims=([2,3],[0,1]))
        # Summing biases
        if self.use_bias:
            output += self.bias
        return output

    def extra_repr(self) -> str:
        return 'in_channels={}, out_channels={}, bias={}'.format(
            self.in_channels, self.out_channels, self.use_bias
        )

def multipleBFS(adj: coo_matrix, kernel_size: int, hops: int) -> torch.LongTensor:
    r"""Computes nodes neighborhood for a given graph using multiple calls to Breath-First Search algorithm.

    Keyword args:
    adj -- scipy.sparse.coo_matrix, adjacency matrix described as coo_matrix
    kernel_size -- int, size of nodes neighborhoods
    hops -- int, search depth

    Returns:
    neighborhoods -- torch.LongTensor, dim: |V|*kernel_size
    """
    # Creating networkx graph from adjacency matrix as input
    G = nx.convert_matrix.from_scipy_sparse_array(adj)

    # Defining number of nodes
    V = G.number_of_nodes()

    # Initialize neighborhoods data structure
    neighborhoods = []

    # Computing neighborhoods using multiple BFS
    for x in range(V):
        # Adding self loops
        neighborhoods.append(x)
        adj_counts = 1

        # Breath-First Search
        visited_nodes = list(nx.bfs_edges(G, source=x, depth_limit=hops))

        for _, t in visited_nodes:
            neighborhoods.append(t)
            adj_counts += 1
            if(adj_counts == kernel_size):
                break
        # Adding padding
        while(adj_counts < kernel_size):
            neighborhoods.append(V)
            adj_counts += 1

    return torch.LongTensor(neighborhoods)