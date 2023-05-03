""" Process dataset with HeteroData and a demo of HeterGNN network

TODO:
* HPS:
    * lr: 1e-3
    * weight_decay: 1e-4
    * hidden_size: 128
    * num_heads: 2
    * num_conv_layers: 2
    * epochs: 200
    * batch_size: 64
* replace `HANConv` with `HGTConv`

"""

import argparse
import os.path as osp
import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torch.nn import CrossEntropyLoss, MSELoss
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from py_script.dataset import GMD, MultiGMD
from py_script.model import HGT
from py_script.transforms import NormalizeColumnFeatures

torch.manual_seed(12345)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--names", type=str, default=["epri21"], nargs='+',
                        help="list of names of networks")
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
    parser.add_argument("--num_conv_layers", type=int, default=1,
                        help="number of layers in HGT")
    parser.add_argument("--num_mlp_layers", type=int, default=4,
                        help="number of layers in MLP")
    activation_choices = ["relu", "rrelu", "hardtanh", "relu6", "sigmoid", "hardsigmoid", "tanh", "silu",
                          "mish", "hardswish", "elu", "celu", "selu", "glu", "gelu", "hardshrink",
                          "leakyrelu", "logsigmoid", "softplus", "tanhshrink"]
    parser.add_argument("--activation", type=str, default="relu", choices=activation_choices,
                        help="specify the activation function used")
    parser.add_argument("--conv_type", type=str, default="hgt", choices=["hgt", "han"],
                        help="select the type of convolutional layer (hgt or han)")
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
    parser.add_argument("--processor", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="selects between CPU or CUDA")
    parser.add_argument("--weight", action="store_true",
                        help="use weighted loss.")
    parser.add_argument("--setting", "-s", type=str, default="gic", choices=["mld", "gic"],
                        help="Specify the problem setting, either `mld` or `gic`")
    args = vars(parser.parse_args())

    if args['normalize']:
        pre_transform = T.Compose([NormalizeColumnFeatures(['x', 'edge_attr'])])
    else:
        pre_transform = None

    ROOT = osp.join(osp.expanduser('~'), 'tmp', 'data', 'GMD')
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Select the processor to use
    if args['processor'] == "cpu":
        DEVICE = torch.device('cpu')
    elif args['processor'] == "cuda":
        DEVICE = torch.device('cuda')
    elif args['processor'] == "auto":
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        print("Unknown processor type: " + args['processor'] + ". Defaulting to \"auto\".")
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if len(args['names']) == 1:
        dataset = GMD(ROOT,
                  name=args['names'][0],
                  setting=args['setting'],
                  problem=args['problem'],
                  force_reprocess=args['force'],
                  pre_transform=pre_transform)
    elif len(args['names']) > 1:
        dataset = MultiGMD(ROOT,
                  names=args['names'],
                  setting=args['setting'],
                  problem=args['problem'],
                  force_reprocess=args['force'],
                  pre_transform=pre_transform)
    else:
        raise "Please input at least one grid name"
    data = dataset[0]

    # Train and test split for our datasets
    dataset_train, dataset_test = train_test_split(dataset,
                                                   test_size=args['test_split'],
                                                   random_state=12345,
                                                   shuffle=True)
    dataset_train, dataset_val = train_test_split(dataset_train,
                                                  test_size=args['test_split'],
                                                  random_state=12345,
                                                  shuffle=True)

    # Create a DataLoader for our datasets
    data_loader_train = DataLoader(dataset=dataset_train,
                                   batch_size=args['batch_size'],
                                   shuffle=True)
    data_loader_val = DataLoader(dataset=dataset_val,
                                 batch_size=1,
                                 shuffle=True)
    data_loader_test = DataLoader(dataset=dataset_test,
                                  batch_size=1,
                                  shuffle=True)

    # adjust the output dimension accordingly
    out_channels = 2 if args['problem'] == "clf" else 1
    model = HGT(hidden_channels=args['hidden_size'],
                num_mlp_layers=args['num_mlp_layers'],
                conv_type=args['conv_type'],
                activation=args['activation'],
                out_channels=out_channels,
                num_conv_layers=args['num_conv_layers'],
                num_heads=args['num_heads'],
                dropout=args['dropout'],
                node_types=data.node_types,
                metadata=data.metadata()
                ).to(DEVICE)

    # adjust the loss function accordingly
    loss_fn = CrossEntropyLoss() if args['problem'] == "clf" else MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args['lr'],
                                 weight_decay=args['weight_decay'])

    losses = []
    if len(args['names']) == 1:
        pbar = tqdm(range(args['epochs']), desc=args['names'][0])
    else:
        pbar = tqdm(range(args['epochs']), desc="Multi Grids")
    model.train()
    for epoch in pbar:
    # for epoch in range(args['epochs']):
        t_loss = 0
        for i, data in enumerate(data_loader_train, 0):
            data = data.to(DEVICE)
            optimizer.zero_grad()

            # Decide between MLD or GIC
            if args['setting'] == "mld":
                out = model(data)[data.load_bus_mask]
                loss = F.mse_loss(out, data['y'])
            else:
                out = model(data, "gmd_bus")
                if args['weight']:
                    weight = len(data['y']) / (2 * data['y'].bincount())
                    loss = F.cross_entropy(out, data['y'], weight=weight)
                else:
                    loss = F.cross_entropy(out, data['y'])
                
                train_acc = (data['y'].detach().cpu().numpy() == out.argmax(
                dim=1).detach().cpu().numpy()).sum() / len(data['y'])
                roc_auc = roc_auc_score(data['y'].detach().cpu().numpy(), out.argmax(1).detach().cpu().numpy())
            loss.backward()
            optimizer.step()

            # FIXED: we don't need to devide by num_graphs
            t_loss += loss.item()
        
        # Choose how to handle the pbar based on the problem setting
        if args['setting'] == "mld":
            pbar.set_postfix({"loss": t_loss})
        else:
            pbar.set_postfix({"loss": t_loss, "train_acc": train_acc, "roc_auc": roc_auc})
        losses.append(t_loss)

    # exit()
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

    # Choose how to handle the figure based on the problem setting
    if args['setting'] == "mld":
        plt.title(f"Hete-Graph - {args['problem']}")
        plt.savefig(f"Figures/Losses/losses - {args['problem']}_{losses_count}_final-t_loss={t_loss}_.png")

        # Evaluate the model
        plt.clf()
        model.eval()
        for data in data_loader_test:
            pred = model(data)[data.load_bus_mask]
            plt.plot(data['y'], "r.", label="true")
            loss = F.mse_loss(pred, data['y'])
            print(loss.item())
            plt.plot(pred.detach().cpu().numpy(), "b.", label="pred")
            plt.legend()
            plt.savefig(f"Figures/Predictions/result_{args['problem']}_{predictions_count}.png")
            exit()
    else:
        if args['weight']:
            plt.title(f"weighted loss: {t_loss:.4f}"
                    + "\n"
                    + f"accuracy: {train_acc:.4f}"
                    + "\n"
                    + f"ROC_AUC score: {roc_auc:.4f}")
        else:
            plt.title(f"loss: {t_loss:.4f}"
                    + "\n"
                    + f"accuracy: {train_acc:.4f}"
                    + "\n"
                    + f"ROC_AUC score: {roc_auc:.4f}")
        # plt.savefig(f"Figures/Losses/losses - {args['problem']}_{losses_count}_final-t_loss={t_loss}_.png")
        plt.savefig("GIC-loss.png")
