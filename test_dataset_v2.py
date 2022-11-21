""" Process dataset with HeteroData and a demo of HeterGNN network """

import torch
from torch_geometric.nn import HGTConv, Linear
from torch import nn
from py_script.dataset_v2 import GMD


class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return self.lin(x_dict['bus'])


dataset = GMD("./test/data", name="b4gic")
print(dataset)
data = dataset[0]

model = HGT(hidden_channels=64, out_channels=dataset.num_classes, num_heads=2, num_layers=2)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    loss = loss_fn(out, data['y'])
    loss.backward()
    optimizer.step()
    print(f"epoch {epoch}, loss {loss:.4f}")
