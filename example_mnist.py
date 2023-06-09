"""
This file contains an example of application of the DGCF layer on the MNIST dataset.
Images were converted in graphs as reported in on the paper.
"""

from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from DGCF import DGCF
from copy import deepcopy
from utils import ConvGNN, multipleBFS
from scipy.sparse import coo_matrix
import numpy as np
import torch
import torch.nn as nn

# hyperparameters definition
N_EPOCHS = 10
BATCH_SIZE = 128
KERNEL_SIZE = 6
N_CLASSES = 10
CORR_THRESHOLD = 0.5
IN_CHANNELS = 1
OUT_CHANNELS = 4
OUT_CHANNELS_STATIC_CONV = 20
FGN_HIDDEN_NODES = 200
LR = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# dataset loading
print("Loading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train.reshape(60000, 784), X_test.reshape(10000, 784)
X_train, X_test = X_train.astype('float32'), X_test.astype('float32')
X_train /= 255; X_test /= 255 # normalizing in [0, 1]

# graph extraction from images
print("Graph extraction from MNIST...")
ix = np.where(np.var(X_train, axis=0) > 0)[0]                       # exclusion of constant pixels
X_train, X_test = X_train[:,ix], X_test[:,ix]
corr_coeff = np.abs(np.corrcoef(X_train.transpose()))               # correlation matrix
adj = coo_matrix((corr_coeff > CORR_THRESHOLD).astype(int))         # correlation thresholding
neighborhoods = multipleBFS(adj, kernel_size=KERNEL_SIZE, hops=4)   # BFS for neighborhood search
neighborhoods = neighborhoods.reshape(-1, KERNEL_SIZE).numpy()
X_train, X_test = np.expand_dims(X_train, axis=-1), np.expand_dims(X_test, axis=-1)
N_NODES = X_train.shape[1]
print(f"MNIST graph shape (n_nodes, in_channels): ({N_NODES}, {X_train.shape[-1]})")

# train/validation split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, stratify=y_train)
X_train, y_train = torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()
X_val, y_val = torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long()
X_test, y_test = torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long()
print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")
print(f"Test set shape: {X_test.shape}")

# DataLoader creation
training_dataloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train, y_train), shuffle=True, batch_size=BATCH_SIZE)
validation_dataloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_val, y_val), shuffle=False, batch_size=BATCH_SIZE)
test_dataloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_test, y_test), shuffle=False, batch_size=BATCH_SIZE)

# filter-generating network definition
filter_generating_network = nn.Sequential(
    torch.nn.Linear(IN_CHANNELS * N_NODES, FGN_HIDDEN_NODES),
    torch.nn.ReLU(),
    torch.nn.Linear(FGN_HIDDEN_NODES, OUT_CHANNELS * IN_CHANNELS * KERNEL_SIZE),
)

# GNN definition
net = nn.Sequential(
    DGCF(n_nodes=N_NODES, 
        kernel_size=KERNEL_SIZE, 
        neighborhoods=neighborhoods, 
        in_channels=IN_CHANNELS, 
        out_channels=OUT_CHANNELS,
        filter_generating_network=filter_generating_network),
    nn.ReLU(),
    ConvGNN(n_nodes=N_NODES, 
        kernel_size=KERNEL_SIZE, 
        neighborhoods=neighborhoods, 
        in_channels=OUT_CHANNELS, 
        out_channels=OUT_CHANNELS_STATIC_CONV),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(N_NODES * 20, N_CLASSES)
)
net = net.to(DEVICE)
print("GNN network:")
print(net)

# training loop
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
best_loss = float('+inf')
best_net = None
print("Training stage:")
for epoch in range(N_EPOCHS):
    net.train()
    running_training_loss = .0
    # training set
    for (x, y) in training_dataloader:
        # forward pass
        optimizer.zero_grad()
        p = net(x.to(DEVICE)).cpu()
        loss = nn.functional.cross_entropy(p, y)
        # backward pass
        loss.backward()
        optimizer.step()
        running_training_loss += loss.item()
    running_training_loss /= len(training_dataloader)

    net.eval()
    running_val_loss = .0
    # validation set
    with torch.no_grad():
        for (x, y) in validation_dataloader:
            p = net(x.to(DEVICE)).cpu()
            loss = nn.functional.cross_entropy(p, y)
            running_val_loss += loss.item()
        running_val_loss /= len(validation_dataloader)

    print(f'Epoch {epoch}\tTraining loss: {running_training_loss:.5f}\tValidation loss: {running_val_loss:.5f}')

    # model update
    if running_val_loss < best_loss:
        best_loss = running_val_loss
        best_net = deepcopy(net)
print("Training completed.")
net = best_net
net.eval()

# computing accuracy of training/validation/test
@torch.no_grad()
def accuracy(net: nn.Sequential, dataloader: torch.utils.data.DataLoader) -> float:
    correct = 0
    n_samples = 0
    for (x, y) in dataloader:
        p = net(x.to(DEVICE)).cpu()
        correct += (p.argmax(1).flatten() == y).float().sum()
        n_samples += len(x)
    return (correct / n_samples)*100

print(f'Training accuracy: {accuracy(net, training_dataloader):.5f}') 
print(f'Validation accuracy: {accuracy(net, validation_dataloader):.5f}') 
print(f'Test accuracy: {accuracy(net, test_dataloader):.5f}') 
