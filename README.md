# Dynamic Graph Convolutional Filters

PyTorch implementation of the Dynamic Graph Convolutional Filter (DGCF) layer presented in <i>"Adaptive Filters in Graph Convolutional Neural Networks"</i> (<a href="https://arxiv.org/pdf/2105.10377.pdf">arXiv preprint arXiv:2105.10377</a>)

Params:
- `n_nodes`, int: number of nodes of the input graphs
- `kernel_size`, int: number of neighbors for convolution
- `neighborhoods`, torch.Tensor: matrix $N$ with dimensions (n_nodes, kernel_size) where the entry $N_{ij}$
        denotes for the $i$'s $j$-th closest neighbor.
- `in_channels`, int: number of input channels for each node
- `out_channels`, int: number of output channels for each node
- `filter_generating_network`, nn.Module: filter generating network. It receives in input a feature vector resulting by
        the vectorization of all the nodes' features. It returns the vectorized weights 
        (whose dimension is out_channels * in_channels * kernel_size [+ out_channels if bias is dynamic]).
- `bias`, string (default, 'static'):, whether the layer uses a bias vector. It can be:
- - `static`, whether the layer uses a single bias vector for each input
- - `dynamic`, whether the biases are generated from the filter_generating_network. In such case, the biases will be determined by the last [+ out_channels] weights generated from the filter_generating network.
- `bias_initializer`, Callable: Initializer for the bias vector (zeros by default. Use nn.init.* inizializations)

Input shape: `(batch_size, n_nodes, in_channels)`
Output shape: `(batch_size, n_nodes, out_channels)`

Example of usage:
``` python
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
```
