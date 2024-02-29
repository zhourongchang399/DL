import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.datasets import KarateClub

# # Define a simple graph for demonstration
# edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
# x = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.float)

data = KarateClub().data

# # Create a PyTorch Geometric Data object
# data = Data(x=x, edge_index=edge_index)

# Define a simple Graph Convolutional Network (GCN)
class GCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Apply the Graph Convolutional Layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

model = GCN(34, 4)  
criterion = torch.nn.NLLLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  
  
model.train()  
for epoch in range(200):  
    optimizer.zero_grad()  
    out = model(data)  
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  
    loss.backward()  
    optimizer.step()
