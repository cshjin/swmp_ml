from py_script.dataset import GMD
import torch_geometric.utils as pygutils
from torch_geometric.nn import SAGEConv, GCNConv
import torch
from torch import nn
# Simple GNN model


class GNN(torch.nn.Module):

    def __init__(self, in_channel=35, out_channel=1):
        super(GNN, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv1 = SAGEConv(in_channel, 128)
        self.conv2 = SAGEConv(128, out_channel)
        self.activation = torch.tanh

    def forward(self, data):

        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        # print(x.shape)
        x = self.activation(x)
        x = self.conv2(x, edge_index)
        x = self.activation(x)

        return x


if __name__ == "__main__":
    dataset = GMD("./test/data", name="b4gic")
    print(dataset)
    data = dataset[0]
    model = GNN(in_channel=data.num_node_features, out_channel=1)
    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Optimizer: Adam usually works well
    losses = []
    batch_size = 4  # Needs to be tuned
    for i in range(100):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {i+1}, MSE loss {loss:.4f}")
