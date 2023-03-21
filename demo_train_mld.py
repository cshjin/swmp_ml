""" Process dataset with HeteroData and a demo of HeterGNN network

TODO:
* HPS:
    * lr: 1e-3
    * weight_decay: 1e-4
    * hidden_size: 128
    * num_heads: 2
    * num_layers: 2
    * epochs: 200
    * batch_size: 64
* replace `HANConv` with `HGTConv`

"""

import argparse
import os.path as osp

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss, MSELoss
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from py_script.dataset import GMD, MultiGMD
from py_script.model import HGT
from py_script.transforms import NormalizeColumnFeatures

torch.manual_seed(12345)


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
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout rate")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of epochs in training")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="batch size in training")
    parser.add_argument("--normalize", action="store_true",
                        help="normalize the data")
    parser.add_argument("--test_split", type=float, default=0.2,
                        help="the proportion of datasets to use for testing")
    args = vars(parser.parse_args())

    if args['normalize']:
        pre_transform = T.Compose([NormalizeColumnFeatures(['x', 'edge_attr'])])
    else:
        pre_transform = None

    ROOT = osp.join(osp.expanduser('~'), 'tmp', 'data', 'GMD')
    if args['name'] != "all":
        dataset = GMD(ROOT,
                      name=args['name'],
                      problem=args['problem'],
                      force_reprocess=args['force'],
                      pre_transform=pre_transform)
        data = dataset[0]
    else:
        dataset = MultiGMD(ROOT,
                           name=args['name'],
                           problem=args['problem'],
                           force_reprocess=args['force'],
                           pre_transform=pre_transform)

    # Train and test split for our datasets
    dataset_train, dataset_test = train_test_split(dataset, test_size=args['test_split'], random_state=12345)

    # Create a DataLoader for our datasets
    data_loader_train = DataLoader(dataset=dataset_train,
                                   batch_size=args['batch_size'],
                                   shuffle=True)

    data_loader_test = DataLoader(dataset=dataset_test,
                                  batch_size=args['batch_size'],
                                  shuffle=True)

    # adjust the output dimension accordingly
    out_channels = 2 if args['problem'] == "clf" else 1
    model = HGT(hidden_channels=args['hidden_size'],
                out_channels=out_channels,
                num_heads=args['num_heads'],
                num_layers=args['num_layers'],
                dropout=args['dropout'],
                node_types=data.node_types,
                metadata=data.metadata()
                )

    # adjust the loss function accordingly
    loss_fn = CrossEntropyLoss() if args['problem'] == "clf" else MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args['lr'],
                                 weight_decay=args['weight_decay'])

    losses = []
    pbar = tqdm(range(args['epochs']), desc=args['name'])
    model.train()
    for epoch in pbar:
        t_loss = 0
        for i, data in enumerate(data_loader_train, 0):
            optimizer.zero_grad()
            out = model(data)
            loss = F.mse_loss(out, data['y'])
            loss.backward()
            optimizer.step()

            t_loss += loss.item() / data.num_graphs
        pbar.set_postfix({"loss": t_loss})
        losses.append(t_loss)

    exit()
    ''' plot the loss function '''
    fig = plt.figure(figsize=(4, 3), tight_layout=True)
    foo = r'training loss'
    plt.plot(losses)
    plt.ylabel(foo)
    plt.xlabel("epoch")
    plt.title(f"Hete-Graph - {args['problem']}")
    plt.savefig(f"losses - {args['problem']}.png")

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
        plt.savefig("tmp.png")
        exit()