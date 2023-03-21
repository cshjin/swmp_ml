""" Process dataset with HeteroData and a demo of HeterGNN network """

import argparse
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch_geometric.transforms.normalize_features as normalize
from scipy.special import softmax
from sklearn import datasets
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.preprocessing import normalize
# from torch import nn
from torch.nn import (CrossEntropyLoss, Dropout, Module, ModuleDict,
                      ModuleList, MSELoss, ReLU, Sequential)
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.loader import DataLoader
# from torch.utils.data import Dataset
from torch_geometric.nn import HANConv, HGTConv, Linear
from torch_geometric.transforms import BaseTransform
from tqdm import tqdm

from py_script.dataset import GMD, MultiGMD

import os, os.path

class NormalizeColumnFeatures(BaseTransform):
    r"""Row-normalizes the attributes given in :obj:`attrs` to sum-up to one
    (functional name: :obj:`normalize_features`).

    Args:
        attrs (List[str]): The names of attributes to normalize.
            (default: :obj:`["x"]`)
    """

    def __init__(self, attrs: List[str] = ["x"]):
        self.attrs = attrs

    def __call__(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                value = value - value.min()
                value.div_(value.sum(dim=0, keepdim=True).clamp_(min=1.))
                store[key] = value
        return data


class HGT(Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.num_heads = num_heads
        self.num_layers = num_layers

        self.lin_dict = ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = ModuleList()
        for _ in range(self.num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(), self.num_heads, group='min')
            # conv = HANConv(hidden_channels, hidden_channels, data.metadata(), self.num_heads, dropout=0)
            self.convs.append(conv)

        self.flatten = Sequential(
            Linear(hidden_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, out_channels),
        )
        # self.lin = Linear(hidden_channels, out_channels)

    def forward(self, data):
        # Some small error checks:
        # The number of entries in aligned_keys should be the same as the number of entries in num_nodes
        # if(not(len(aligned_keys) == len(num_nodes))):
        #     print("Dimension mismatch between the aligned keys and number of nodes (forward function).")
        # exit()
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        node_idx = data.node_idx_y

        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        # DEBUG: x_dict['gen'] is None
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        batch_node_idx = []
        for s in range(data.num_graphs):
            for idx in node_idx[0]:
                batch_node_idx.append(idx + 19 * s)

        # batch_node_idx = [node_idx[0] + 19 * s for s in range(data.num_graphs)]
        output = x_dict['bus'][batch_node_idx]  # dim: 320 * 128
        return self.flatten(output)  # dim: 320 * 1


if __name__ == "__main__":

    # TODO: 
    # HPS: 
    # * lr: 1e-3
    # * weight_decay: 1e-4
    # * hidden_size: 128
    # * num_heads: 2
    # * num_layers: 2
    # * epochs: 200
    # * batch_size: 64



    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="epri21",
                        help="name of network")
    parser.add_argument("--problem", "-p", type=str, default="clf", choices=["clf", "reg"],
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
    parser.add_argument("--batch_size", type=int, default=64,
                        help="batch size in training")
    parser.add_argument("--pre_transform", type=str, default="normalize", choices=["normalize"],
                        help="the transform function used while processing the data")
    parser.add_argument("--test_split", type=float, default=0.2,
                        help="the proportion of datasets to use for testing")
    args = vars(parser.parse_args())

    # Get the type of pre-transform function
    if(args['pre_transform'] == "normalize"):
        pre_transform_function = NormalizeColumnFeatures()
    else:
        pre_transform_function = None
    # pre_transform_function = normalize

    # readout the dataset `pyg.heterodata`
    if args['name'] != "all":
        dataset = GMD("./test/data",
                      name=args['name'],
                      problem=args['problem'],
                      force_reprocess=args['force'],
                      pre_transform=T.Compose([NormalizeColumnFeatures(['x', 'edge_attr'])]))
        data = dataset[0]
    else:
        dataset = MultiGMD("./test/data",
                           name=args['name'],
                           problem=args['problem'],
                           force_reprocess=args['force'],
                           pre_transform=pre_transform_function)

    # Train and test split for our datasets
    dataset_train, dataset_test = train_test_split(dataset, test_size=args['test_split'])

    # Create a DataLoader for our datasets
    data_loader_train = DataLoader(dataset=dataset_train,
                                   batch_size=64,
                                   shuffle=True)

    data_loader_test = DataLoader(dataset=dataset_test,
                                  batch_size=64,
                                  shuffle=True)

    # adjust the output dimension accordingly
    out_channels = 2 if args['problem'] == "clf" else 1
    model = HGT(hidden_channels=args['hidden_size'],
                out_channels=out_channels,
                num_heads=args['num_heads'],
                num_layers=args['num_layers'])

    # adjust the loss function accordingly
    loss_fn = CrossEntropyLoss() if args['problem'] == "clf" else MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args['lr'],
                                 weight_decay=args['weight_decay'])
    # optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])

    # Training code for the DataLoader version
    losses = []  # Create a list to store the losses each iteration
    pbar = tqdm(range(args['epochs']), desc=args['name'])
    for epoch in pbar:
        # Mini-batch settings for multi-graphs
        t_loss = 0
        for i, data in enumerate(data_loader_train, 0):
            # Get the output from the model and compute the loss
            optimizer.zero_grad()   # Zero the gradient
            out = model(data)
            loss = F.mse_loss(out, data['y'])

            # Update the gradient and use it
            loss.backward()        # Perform backpropagation
            optimizer.step()        # Update the weights

            # Add the loss of the current iteration to the total for the epoch
            t_loss += loss.item()

            # Print some information about the current iteration
            # print("Epoch", epoch, "i", i, "loss.item()", loss.item())
        pbar.set_postfix({"loss": t_loss})

        # Store some information about the accumulated loss in the current
        # iteration of the epoch
        losses.append(t_loss)

    # Count the number of files that exist in the Figures directory, so
    # we can give a unique name to the two new figures we're creating
    losses_count = len([file_name for file_name in os.listdir('./Figures/Losses/')])
    predictions_count = len([file_name for file_name in os.listdir('./Figures/Predictions/')])

    ''' plot the loss function '''
    fig = plt.figure(figsize=(4, 3), tight_layout=True)
    foo = r'training loss'
    plt.plot(losses)
    plt.ylabel(foo)
    plt.xlabel("epoch")
    plt.title(f"Hete-Graph - {args['problem']}")
    plt.savefig(f"Figures/Losses/losses - {args['problem']}_{losses_count}_final-t_loss={t_loss}_.png")

    # Evaluate the model
    plt.clf()
    model.eval()
    for data in data_loader_test:
        pred = model(data)
        plt.plot(data['y'], "r.", label="true")
        loss = F.mse_loss(pred, data['y'])
        print(loss.item())
        plt.plot(pred.detach().cpu().numpy(), "b.", label="pred")
        plt.legend()
        plt.savefig(f"Figures/Predictions/result_{args['problem']}_{predictions_count}.png")
        exit()

# TODO:
# - python.exe .\demo_train_mlp.py --problem reg --force --num_layers 3 --test_split 0.2 --hidden_size 64 --epochs 500


# TODO:
# check without HeteroGNN, use MLP(data.x_dict['bus']) instead.
# Try out batch normalization
