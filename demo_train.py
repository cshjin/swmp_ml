""" Process dataset with HeteroData and a demo of HeterGNN network """

import argparse

import matplotlib.pyplot as plt
import torch
from scipy.special import softmax
from torch import nn
from torch_geometric.nn import HGTConv, Linear

from py_script.dataset import GMD, MultiGMD
from tqdm import tqdm


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

        return self.lin(x_dict['gmd_bus'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="epri21",
                        help="name of network")
    parser.add_argument("--problem", "-p", type=str, default="reg", choices=["clf", "reg"],
                        help="Specify the problem, either `clf` or `reg`")
    parser.add_argument("--force", action="store_true",
                        help="Force to reprocess data")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="weight decay rate for Adam")
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="hidden dimension in HGT")
    parser.add_argument("--num_heads", type=int, default=2,
                        help="number of heads in HGT")
    parser.add_argument("--num_layers", type=int, default=1,
                        help="number of layers in HGT")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of epochs in training")
    parser.add_argument("--log", action="store_true",
                        help="Log the training process.")
    args = vars(parser.parse_args())

    # readout the dataset `pyg.heterodata`
    if args['name'] != "all":
        dataset = GMD("./test/data",
                      name=args['name'],
                      problem=args['problem'],
                      force_reprocess=args['force'])
        data = dataset[0]
        # print(data['bus'].x.shape,
        #       data['gen'].x.shape,
        #       data['gmd_bus'].x.shape)
        # print(data['branch'].edge_attr.shape,
        #       data['branch_gmd'].edge_attr.shape,
        #       data['gmd_branch'].edge_attr.shape)
        # print(data[("gen", "conn", "bus")].edge_index.shape)
        # print(data[("gmd_bus", "attach", "bus")].edge_index.shape)

        # exit()
    else:
        dataset = MultiGMD("./test/data",
                           name=args['name'],
                           problem=args['problem'],
                           force_reprocess=args['force'])
        data = dataset[0]
    print(dataset)

    # adjust the output dimension accordingly
    out_channels = 2 if args['problem'] == "clf" else 1
    model = HGT(hidden_channels=args['hidden_size'],
                out_channels=out_channels,
                num_heads=args['num_heads'],
                num_layers=args['num_layers'])

    # adjust the loss function accordingly
    loss_fn = nn.CrossEntropyLoss() if args['problem'] == "clf" else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    losses = []
    model.train()
    pbar = tqdm(range(args['epochs']))
    for epoch in pbar:
        optimizer.zero_grad()
        # data.edge_attr_dict
        # mini-batch settings for multi-graphs
        out = model(data.x_dict, data.edge_index_dict)
        if args['problem'] == "reg":
            # REVIEW: meet the dimension of target
            out = out.T[0]
        loss = loss_fn(out, data['y'])
        loss.backward()
        optimizer.step()
        pbar.set_postfix({'loss': loss.item()})
        losses.append(loss.item())

    print("True bic_placed", data.y)
    out = out.detach().cpu().numpy()
    if args['problem'] == "clf":
        print(softmax(out, axis=1))
        print("Pred bic_placed", out.argmax(1))
    else:
        print(out)
    # print("Predicted bic_placed", out.argmax(1))

    ''' plot the loss function '''
    # fig = plt.figure(figsize=(4, 3), tight_layout=True)
    # foo = r'training loss'
    # plt.plot(losses)
    # plt.ylabel(foo)
    # plt.xlabel("epoch")
    # plt.title(f"Hete-Graph - {args['problem']}")
    # plt.savefig(f"losses - {args['problem']}.png")