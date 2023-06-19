""" Demo training script for HGT on GMD datasets """

import os.path as osp
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from py_script.dataset import GMD, MultiGMD
from py_script.model import HGT
from py_script.transforms import NormalizeColumnFeatures
from py_script.utils import create_dir, process_args

torch.manual_seed(12345)

if __name__ == "__main__":
    args = process_args()

    if args['normalize']:
        pre_transform = T.Compose([NormalizeColumnFeatures(['x', 'edge_attr'])])
    else:
        pre_transform = None

    ROOT = osp.join(osp.expanduser('~'), 'tmp', 'data', 'GMD')
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Select the processor to use
    if args['gpu'] < 0 or (not torch.cuda.is_available()) or (args['gpu'] >= torch.cuda.device_count()):
        DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device(f"cuda:{args['gpu']}")

    if len(args['names']) == 1:
        dataset = GMD(ROOT,
                      name=args['names'][0],
                      setting=args['setting'],
                      force_reprocess=args['force'],
                      pre_transform=pre_transform)
    elif len(args['names']) > 1:
        dataset = MultiGMD(ROOT,
                           names=args['names'],
                           setting=args['setting'],
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
    out_channels = 2 if args['setting'] == "gic" else 1
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
    # loss_fn = CrossEntropyLoss() if args['setting'] == "gic" else MSELoss()
    # loss_fn = F.cross_entropy if args['setting'] == "gic" else F.mse_loss
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args['lr'],
                                 weight_decay=args['weight_decay'])

    # Some extra code to check if the optimizer really did place some blockers
    # in the results and if model really is placing some blockers in the grid
    num_optimizer_blockers = 0
    num_model_blockers = 0

    losses = []
    if len(args['names']) == 1:
        pbar = tqdm(range(args['epochs']), desc=args['names'][0])
    else:
        pbar = tqdm(range(args['epochs']), desc="Multi Grids")
    model.train()
    for epoch in pbar:
        t_loss = 0
        for i, data in enumerate(data_loader_train):
            data = data.to(DEVICE)
            optimizer.zero_grad()

            # Decide between MLD or GIC
            if args['setting'] == "mld":
                out = model(data)[data.load_bus_mask]
                loss = F.mse_loss(out, data['y'])
            else:
                out = model(data, "gmd_bus")[data.gic_blocker_bus_mask]
                train_y = data['y'][data.gic_blocker_bus_mask]
                if args['weight'] and (len(train_y.bincount()) > 1):
                    # print(2 * train_y.bincount())
                    # weight = len(train_y) / (2 * train_y.bincount(minlength=2))
                    weight = len(train_y) / (2 * train_y.bincount())
                    loss = F.cross_entropy(out, train_y, weight=weight)
                else:
                    loss = F.cross_entropy(out, train_y)

                train_acc = (train_y.detach().cpu().numpy() == out.argmax(
                    1).detach().cpu().numpy()).sum() / len(train_y)

                # ROC_AUC score can't be computed when there's only one class (all zeroes in the true output)
                if len(train_y.bincount()) > 1:
                    # There is at least one blocket placed, so we can compute the roc_auc score
                    roc_auc = roc_auc_score(train_y.detach().cpu().numpy(), out.argmax(1).detach().cpu().numpy())
                else:
                    # TODO: change the default value of roc_auc. We're supposed to throw an error, though.
                    roc_auc = 0

            loss.backward()
            optimizer.step()

            t_loss += loss.item()

        # Choose how to handle the pbar based on the problem setting
        if args['setting'] == "mld":
            pbar.set_postfix({"loss": t_loss})
        else:
            pbar.set_postfix({"loss": t_loss, "train_acc": train_acc, "roc_auc": roc_auc})
        losses.append(t_loss)

    # Count the number of files that exist in the Figures directory, so
    # we can give a unique name to the two new figures we're creating
    # losses_count = len([file_name for file_name in os.listdir('./Figures/Losses/')])
    # predictions_count = len([file_name for file_name in os.listdir('./Figures/Predictions/')])

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    create_dir("./Figures")

    ''' plot the loss function '''
    fig = plt.figure(figsize=(4, 3), tight_layout=True)
    plt.plot(losses)
    plt.ylabel("training loss")
    plt.xlabel("epoch")
    plt.title(f"Problem: {args['setting'].upper()}\n" +
              f"Grid: {''.join(args['names'])}\n" +
              f"Loss: {t_loss:.4f}")
    # plt.savefig(f"./Figures/{args['setting']}_{ts}.png")

    # exit()
    # Choose how to handle the figure based on the problem setting
    if args['setting'] == "mld":
        plt.title(f"Hete-Graph - {args['setting']}")
        plt.savefig(f"./Figures/{args['setting']}_{ts}_Loss={t_loss}.png")

        # TOFIX: Evaluate the model, the code below is not correct.
        # plt.clf()
        # model.eval()
        # for data in data_loader_test:
        #     pred = model(data)[data.load_bus_mask]
        #     plt.plot(data['y'], "r.", label="true")
        #     loss = F.mse_loss(pred, data['y'])
        #     print("Testing loss: " + str(loss.item()))
        #     plt.plot(pred.detach().cpu().numpy(), "b.", label="pred")
        #     plt.legend()
        #     plt.savefig(f"Figures/Predictions/result_{args['setting']}_{predictions_count}.png")
        #     exit()
    else:
        print("Test results")
        print("Weighted loss: " + str(t_loss))
        print("Accuracy: " + str(train_acc))
        print("ROC_AUC score: " + str(roc_auc))
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
        # plt.savefig(f"Figures/Predictions/result_{args['problem']}_{predictions_count}.png")
        plt.savefig("Figures/GMD (uiuc150) with single objective (test_acc).png")
