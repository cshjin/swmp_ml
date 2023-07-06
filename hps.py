"""TODO:
* test with multiple objectives, return accuracy and ROC-AUC scoore the validation set
"""

# %%
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import TqdmCallback
from deephyper.problem import HpProblem
from deephyper.search.hps import CBO
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from py_script.dataset import GMD, MultiGMD
from py_script.model import HGT
from py_script.transforms import NormalizeColumnFeatures

import operator

torch.manual_seed(12345)


# %% [run]

def train(model, optimizer, loader, epochs=200, **kwargs):
    r""" Training for the model.

    Args:
        model (object): Model to be trained.
        optimizer (torch.optimizer): Optimizer for the model.
        loader (pyg.DataLoader): Train DataLoader for the model.
        epoch (int, optional): Number of epochs. Defaults to 200.

    Returns:
        tuple: (list of accuracies, list of rocs, list of losses)
    """
    _setting = kwargs.get("setting", "gic")
    _verbose = kwargs.get("verbose", False)

    # return a set of lists
    all_acc, all_roc_auc, all_loss = [], [], []

    if _verbose:
        pbar = tqdm(range(epochs), desc="Training", leave=False)
    else:
        pbar = range(epochs)

    model.train()
    for _ in pbar:
        t_loss = 0
        all_true_labels, all_pred_labels = [], []
        for i, data in enumerate(loader):

            optimizer.zero_grad()

            # Decide between MLD (reg) or GIC (cls)
            if _setting == "mld":
                out = model(data)[data.load_bus_mask]
                loss = F.mse_loss(out, data['y'])
            else:
                out = model(data, "gmd_bus")

                # Apply weighted cross entropy loss
                if weight_arg and (len(data['y'].bincount()) > 1):
                    weight = len(data['y']) / (2 * data['y'].bincount())
                    loss = F.cross_entropy(out, data['y'], weight=weight)
                else:
                    loss = F.cross_entropy(out, data['y'])

                y_true_batch = data['y'].detach().cpu().numpy()
                y_pred_batch = out.argmax(dim=1).detach().cpu().numpy()
                all_true_labels.extend(y_true_batch)
                all_pred_labels.extend(y_pred_batch)

            # update loss
            t_loss += loss.item()
            # update parameters
            loss.backward()
            optimizer.step()

        all_true_labels = np.array(all_true_labels)
        all_pred_labels = np.array(all_pred_labels)

        train_acc = (all_true_labels == all_pred_labels).sum() / len(all_true_labels)
        train_roc_auc = roc_auc_score(all_true_labels, all_pred_labels)

        all_acc.append(train_acc)
        all_roc_auc.append(train_roc_auc)
        all_loss.append(t_loss)

        if _verbose:
            pbar.set_postfix({"loss": t_loss, "acc": train_acc, "roc_auc": train_roc_auc})

    return all_acc, all_roc_auc, all_loss


def evaluate(model, loader, **kwargs):
    r""" Evaluate the model.

    Args:
        model (object): Model to be trained.
        loader (pyg.DataLoader): DataLoader to evaluate the model.

    Returns:
        tuple: (accuracy, roc_auc, loss)
    """
    model.eval()
    _setting = kwargs.get("setting", "gic")

    t_loss = 0
    all_true_labels, all_pred_labels = [], []
    for i, data in enumerate(loader):

        # Decide between MLD (reg) or GIC (cls)
        if _setting == "mld":
            out = model(data)[data.load_bus_mask]
            loss = F.mse_loss(out, data['y'])
        else:
            out = model(data, "gmd_bus")

            # Apply weighted cross entropy loss
            if weight_arg and (len(data['y'].bincount()) > 1):
                weight = len(data['y']) / (2 * data['y'].bincount())
                loss = F.cross_entropy(out, data['y'], weight=weight)
            else:
                loss = F.cross_entropy(out, data['y'])

            y_true_batch = data['y'].detach().cpu().numpy()
            y_pred_batch = out.argmax(dim=1).detach().cpu().numpy()
            all_true_labels.extend(y_true_batch)
            all_pred_labels.extend(y_pred_batch)

        # update loss
        t_loss += loss.item()

    all_true_labels = np.array(all_true_labels)
    all_pred_labels = np.array(all_pred_labels)

    acc = (all_true_labels == all_pred_labels).sum() / len(all_true_labels)
    roc_auc = roc_auc_score(all_true_labels, all_pred_labels)

    return acc, roc_auc, t_loss


def run(config):
    r""" The black-box function for HPS to maximize.

    Args:
        config (dict): The configuration of the hyperparameters.

    Returns:
        float: the objective value to be maximized.

    References:
        * [DeepHyper](https://deephyper.readthedocs.io/en/latest/tutorials/tutorials/colab/HPS_basic_classification_with_tabular_data/notebook.html#Define-the-run-function)

    Notes:
        * The run function is the black-box objective to __maximize__.
    """
    # HPS for model:
    batch_size = config.get("batch_size", 64)
    conv_type = config.get("conv_type", "hgt")
    hidden_channels = config.get("hidden_channels", 64)
    num_mlp_layers = config.get("num_mlp_layers", 64)
    activation = config.get("activation", "relu")
    num_heads = config.get("num_heads", 2)
    num_layers = config.get("num_layers", 1)
    dropout = config.get("dropout", 0.5)
    # HPS for optimizer:
    lr = config.get("lr", 1e-3)
    weight_decay = config.get("weight_decay", 1e-4)

    # NOTE: validate the hps settings

    # Create a DataLoader for our datasets
    loader_train = DataLoader(dataset=dataset_train,
                              batch_size=batch_size,
                              shuffle=True)

    loader_val = DataLoader(dataset=dataset_val,
                            batch_size=batch_size,
                            shuffle=True)

    model = HGT(hidden_channels=hidden_channels,
                conv_type=conv_type,
                num_mlp_layers=num_mlp_layers,
                activation=activation,
                out_channels=2,
                num_heads=num_heads,
                num_conv_layers=num_layers,
                dropout=dropout,
                node_types=data.node_types,
                metadata=data.metadata(),
                ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)

    # training
    train_accs, train_roc_aucs, train_losses = train(model, optimizer, loader_train, verbose=False)
    # evaluate on validation set
    acc, roc_auc, loss = evaluate(model, loader_val)

    # TODO:
    # train the model with loader_train, and evaluate on loader_val
    # option 1: early stopping
    # best, threshold = 1e4, 5
    # for _ in range(epochs):
        # _, _, _  = train(model, optimizer, loader_train, epochs=1, verbose=False)
        # _, _, loss_val = evaluate(model, loader_val)
        # if loss_val < best:
        #   best = loss_val
        #   threshold = 5
        # else:
        #   threshold -= 1 
        # if threshold < 0:
        #  return metrics on val
    # return metrics

    # option 2: save the best metrics from loader_val
    # for _ in range(epochs):
        #  _, _, _  = train(model, optimizer, loader_train, epochs=1, verbose=False)
        # _, _, loss_val = evaluate(model, loader_val)
        # hps = {}
        # hps[epoch] = [acc, roc, loss] # based on validation
    # return max metric from hps

    # Early stopping
    epochs = 200
    hps = {}
    for epoch in range(epochs):
        _, _, _ = train(model, optimizer, loader_train, epochs=1, verbose=False)
        val_acc, roc_auc, loss_val = evaluate(model, loader_val)
        hps[epoch] = [val_acc, roc_auc, loss_val]
    
    # Return the max val_acc from hps
    return max(hps[k][0] for k in hps)

    # stats = {}
    # stats['a'] = [3000, 100, 100]
    # stats['b'] = [5000, 5000, 3000]
    # stats['c'] = [100, 3000, 5000]
    # print(max(stats.items(), key=operator.itemgetter(1))[0])

    # return roc_auc

    # MOO: (acc, roc_auc)
    # SOO: -loss OR acc OR roc_auc

# %%
ROOT = osp.join(osp.expanduser("~"), "tmp", "data", "GMD")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# REVIEW: why we have to use CPU?
DEVICE = torch.device('cpu')

pre_transform = T.Compose([NormalizeColumnFeatures(["x", "edge_attr"])])
# # REVIEW: why we set None to pre_transform?
# # pre_transform = None

setting = "gic"
weight_arg = True
dataset = GMD(ROOT,
              name="epri21",
              setting=setting,
              force_reprocess=False,
              pre_transform=pre_transform)
# # dataset = MultiGMD(ROOT,
# #                    names=["epri21", "uiuc150"],
# #                    setting=setting,
# #                    force_reprocess=True,
# #                    pre_transform=pre_transform)

data = dataset[0]

# # train/val/test: 0.8/0.1/0.1
dataset_train, dataset_test = train_test_split(dataset, test_size=0.2, random_state=41)
dataset_val, dataset_test = train_test_split(dataset_test, test_size=0.5, random_state=41)


problem = HpProblem()

# HPS for model:
problem.add_hyperparameter([16, 32, 64, 128, 256],
                           "hidden_size", default_value=128)
problem.add_hyperparameter([16, 32, 64, 128, 256],
                           "batch_size", default_value=64)
problem.add_hyperparameter([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           "num_conv_layers", default_value=1)
problem.add_hyperparameter([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           "num_mlp_layers", default_value=1)
activation_choices = ["relu", "sigmoid", "tanh", "elu", "leakyrelu"]
problem.add_hyperparameter(activation_choices,
                           "activation", default_value="relu")
problem.add_hyperparameter(["hgt", "han"],
                           "conv_type", default_value="hgt")
problem.add_hyperparameter([1, 2, 4, 8],
                           "num_heads", default_value=2)
problem.add_hyperparameter([0., 0.1, 0.2, 0.3, 0.4, 0.5],
                           "dropout", default_value=0.5)
# HPS for optimizer:
problem.add_hyperparameter((1e-5, 1e-1, "log-uniform"),
                           "lr", default_value=1e-3)
problem.add_hyperparameter([0., 1e-5, 5e-5, 1e-4],
                           "weight_decay", default_value=0.)
# problem.add_hyperparameter((1.0, 3.0), "x")

thread_kwargs = {"num_workers": 8,
                 "callbacks": [TqdmCallback()]}
ray_kwargs = {"num_cpus": 4,
              "num_gpus": 1,
              "num_cpus_per_task": 1,
              "num_gpus_per_task": 1,
              "callbacks": [TqdmCallback()]}

evaluator = Evaluator.create(run,
                             method="thread",
                             method_kwargs=thread_kwargs
                             )
# evaluator = Evaluator.create(run,
#                              method="ray",
#                              method_kwargs=ray_kwargs
#                              )
print("Number of workers: ", evaluator.num_workers)

search = CBO(problem, evaluator, initial_points=[problem.default_configuration], random_state=42)
# Print all the results
print("All results:")
results = search.search(max_evals=200, )
print(results)
results.to_csv("results.csv")

i_max = results['objective'].argmax()
# DEBUG:
best_hp = results.iloc[i_max][0:-3].to_dict()
print("Best HPS: ", best_hp)

# retrain the model with best HPS setting
best_model = HGT(hidden_channels=best_hp['p:hidden_size'],
                 conv_type=best_hp['p:conv_type'],
                 num_mlp_layers=best_hp['p:num_mlp_layers'],
                 activation=best_hp['p:activation'],
                 out_channels=2,
                 num_heads=best_hp['p:num_heads'],
                 num_conv_layers=best_hp['p:num_conv_layers'],
                 dropout=best_hp['p:dropout'],
                 node_types=data.node_types,
                 metadata=data.metadata(),
                 ).to(DEVICE)
best_optimizer = torch.optim.Adam(best_model.parameters(), lr=best_hp['p:lr'], weight_decay=best_hp['p:weight_decay'])

# Create a DataLoader for our datasets
loader_train = DataLoader(dataset=dataset_train,
                          batch_size=best_hp['p:batch_size'],
                          shuffle=True)

loader_val = DataLoader(dataset=dataset_val,
                        batch_size=best_hp['p:batch_size'],
                        shuffle=True)

loader_test = DataLoader(dataset=dataset_test,
                         batch_size=best_hp['p:batch_size'],
                         shuffle=True)

# TODO: update training with best HPS, stop on t
train_metrics = train(best_model, best_optimizer, loader_train)
# for _ in range(epochs):
        #  _, _, _  = train(model, optimizer, loader_train, epochs=1, verbose=False)
        # _, _, loss_val = evaluate(model, loader_val)
        # hps = {}
        # hps[epoch] = [acc, roc, loss, model.parameters()] # based on validation
# load the model parameters based on best metric on validation
test_metrics = evaluate(best_model, dataset_test)
print(f"Test accuracy: {test_metrics[0]:.4f}",
      f"Test ROC-AUC {test_metrics[1]:.4f}")

# TODO:
# plot the radar chart for the best HPS setting from `test_metrics`
# plot the training curve for the best HPS setting from `train_metrics`


# 1. get the best HPS setting from validation set
# 2. train the model again with best HPS setting
# 3. report the test metrics based on the best HPS

# Print the best result
# best_objective_index = results[:]['objective'].argmin()
# print("Best results:")
# print(results.iloc[best_objective_index][0:-3])  # The last 3 slots don't matter

# best_objective_index = results[:]['objective_0'].argmax()
# print("Best result for -test_loss:")
# print(results.iloc[best_objective_index][0:-3], '\n')

# best_objective_index = results[:]['objective_1'].argmax()
# print("Best results for test_acc:")
# print(results.iloc[best_objective_index][0:-3], '\n')

# best_objective_index = results[:]['objective_2'].argmax()
# print("Best results for roc_auc score:")
# print(results.iloc[best_objective_index][0:-3], '\n')


# # Radar (Spider) plot of the 3 objectives
# # Store the labels for the radar plot
# labels = ["-test_loss", "test_acc", "roc_auc"]

# # Number of variables we're plotting.
# num_vars = len(labels)

# # Spread out the labels evenly
# angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# # Since the plot is a circle, finish the loop
# angles += angles[:1]

# # Plot the figure
# fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# # Helper function to plot each row of the results.


# def add_row_to_plot(row, color):
#     values = results.loc[row, ["objective_0", "objective_1", "objective_2"]].tolist()
#     values += values[:1]
#     ax.plot(angles, values, color=color, linewidth=1, label=row)
#     ax.fill(angles, values, color=color, alpha=0.25)


# # Add each row to the plot
# for row, _ in results.iterrows():
#     add_row_to_plot(row, "blue")

# plt.savefig("HPS Test.png")

# # Fix axis to go in the right order and start at 12 o'clock.
# ax.set_theta_offset(np.pi / 2)
# ax.set_theta_direction(-1)

# # Draw axis lines for each angle and label.
# ax.set_thetagrids(np.degrees(angles[:-1]), labels)

# # Go through labels and adjust alignment based on where
# # it is in the circle.
# for label, angle in zip(ax.get_xticklabels(), angles):
#     if angle in (0, np.pi):
#         label.set_horizontalalignment('center')
#     elif 0 < angle < np.pi:
#         label.set_horizontalalignment('left')
#     else:
#         label.set_horizontalalignment('right')

# Save the results to a CSV file


# https://plotly.com/python/radar-chart/
# https://www.pythoncharts.com/matplotlib/radar-charts/

# %%
# python demo_train.py --force --names epri21 --setting gic
# --activation relu --batch_size 64 --conv_type hgt --dropout 0.5
# --hidden_size 128 --lr 5e-4 --num_conv_layers 1 --num_heads 2
# --num_mlp_layers 1 --weight_decay 1e-4 --epochs 250 --weight
