""" Process dataset with HeteroData and a demo of HeterGNN network """

import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import torch
from scipy.special import softmax
from sklearn.model_selection import LeaveOneOut, train_test_split
# from torch import nn
from torch.nn import (CrossEntropyLoss, Module, ModuleDict, ModuleList,
                      MSELoss, ReLU, Sequential, Dropout)
from torch_geometric.loader import DataLoader
# from torch.utils.data import Dataset
from torch_geometric.nn import HGTConv, Linear, HANConv
from tqdm import tqdm

from py_script.dataset import GMD, MultiGMD


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
            # conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
            #    self.num_heads, group='min')
            conv = HANConv(hidden_channels, hidden_channels, data.metadata(), self.num_heads, dropout=0)
            self.convs.append(conv)

        self.flatten = Sequential(
            Linear(hidden_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels),
            ReLU(),
            # Dropout(0.5),
            Linear(hidden_channels, out_channels),
        )
        # self.lin = Linear(hidden_channels, out_channels)

   # def forward(self, x_dict, edge_index_dict):
    def forward(self, x_dict, edge_index_dict, aligned_keys, num_nodes):
        # Some small error checks:
        # The number of entries in aligned_keys should be the same as the number of entries in num_nodes
        if(not(len(aligned_keys) == len(num_nodes))):
            print("Dimension mismatch between the aligned keys and number of nodes (forward function).")
            exit()

        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        # DEBUG: x_dict['gen'] is None
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        # note: return node of generator
        # return self.lin(torch.nn.Dropout(0.5)(x_dict['gen']))
        # Update the output with aligned keys only
        # return self.flatten(x_dict['bus'])
        # output = [x_dict['bus'][k] for k in sum(aligned_keys, [])]  # sum(aligned_keys, []) will merge
        #                                                     # all the lists in aligned_keys into
        #                                                     # one list.
        # TODO: mapping idx based on aligned keys (4,5,...,19,20,...)
        # output = [x_dict['bus'][k] for k in aligned_keys[0]]
        extract_final_indices = []
        index = 0   # Indicates which index we're in for the aligned_keys list.
        shift = 0   # Stores the next index to start from in aligned_keys
        for index in range(len(aligned_keys)):
            # TODO: starting from the second index (index 1), indices will produce a list of tensors rather
            # than a list of integers. I'm not sure why that happens, but I put the int() to typecast the
            # tensor for now.
            indices = [int(k + shift) for k in aligned_keys[index]]  # Shift the aligned keys in the current network
            extract_final_indices.extend(indices)   # Add the shifted aligned_keys to the end of the list
            shift += num_nodes[index]   # Add the number of nodes of the current network to shift to set
                                        # the starting index for the next iteration through the aligned_keys

        output = x_dict['bus'][extract_final_indices]  # dim: 320 * 128
        return self.flatten(output)  # dim: 320 * 1

# class PowerSystemsDataset(Dataset):
#     def __init__(self):
        
#     def __getitem__(self, index):
    
#     def __len__(self):


if __name__ == "__main__":
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
    # data = dataset[0]
    # print(dataset)
    dataset_train, dataset_test = train_test_split(dataset, test_size=0.2)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    # Training code for the DataLoader version
    losses = [] # Create a list to store the losses each iteration
    pbar = tqdm(range(args['epochs']), desc=args['name'])
    for epoch in pbar:
        # Mini-batch settings for multi-graphs
        t_loss = 0
        for i, data in enumerate(data_loader_train, 0):
            # Get the output from the model and compute the loss
            out = model(data.x_dict, data.edge_index_dict, data.source_ids, data.num_network_nodes)
            loss = loss_fn(out, data['y'])

            # Update the gradient and use it
            optimizer.zero_grad()   # Zero the gradient
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

    # Evaluate the model
    model.eval()
    for data in data_loader_test:
        pred = model(data.x_dict, data.edge_index_dict, data.source_ids, data.num_network_nodes)
        print(pred)
        print(data['y'])
        print((pred - data['y']).T)
        plt.plot(data['y'], "ro", label="true")
        plt.plot(pred.detach().cpu().numpy(), "b.", label="pred")
        plt.savefig("tmp.png")
        exit()


    # loo = LeaveOneOut()
    # for i, (train_idx, test_idx) in enumerate(loo.split(np.arange(len(dataset)))):
    #     # print("\n\n\n\n\nTEST TEST TEST TEST TEST TEST  ", test_idx, "\n\n\n\n\n")
    #     losses = []
    #     model.train()
    #     pbar = tqdm(range(args['epochs']))
    #     for epoch in pbar:
    #         optimizer.zero_grad()
    #         # data.edge_attr_dict
    #         # mini-batch settings for multi-graphs
    #         t_loss = 0
    #         for data in dataset[list(train_idx[:1])]:
    #             out = model(data.x_dict, data.edge_index_dict)
    #             loss = loss_fn(out, data['y'])
    #             loss.backward()
    #             optimizer.step()
    #             t_loss += loss.item()
    #             if args['problem'] == "reg":
    #                 # REVIEW: meet the dimension of target
    #                 out = out.T[0]

    #         pbar.set_postfix({'loss': t_loss})
    #         losses.append(t_loss)

    #     print(f"split {i}", train_idx, test_idx)
    #     model.eval()
    #     # train_data = dataset[train_idx[0]]
    #     # print("true pg", train_data.y.T)
    #     # out = out.detach().cpu().numpy()
    #     # print("pred pg", out.T)
    #     # exit()
    #     #
    #     test_data = dataset[test_idx[0]]
    #     print("True pg", test_data.y.T.cpu().numpy())
    #     out = model(test_data.x_dict, test_data.edge_index_dict)
    #     out = out.detach().cpu().numpy()
    #     print("pred pg", out.T)
    #     # print(softmax(out, axis=1))
    #     # print("Pred bic_placed", out.argmax(1))

    #     # reset parameters
    #     for layer in model.children():
    #         if hasattr(layer, 'reset_parameters'):
    #             layer.reset_parameters()

    ''' plot the loss function '''
    # fig = plt.figure(figsize=(4, 3), tight_layout=True)
    # foo = r'training loss'
    # plt.plot(losses)
    # plt.ylabel(foo)
    # plt.xlabel("epoch")
    # plt.title(f"Hete-Graph - {args['problem']}")
    # plt.savefig(f"losses - {args['problem']}.png")
