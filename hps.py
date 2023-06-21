# %%
import os.path as osp

import numpy as np

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import TqdmCallback
from deephyper.problem import HpProblem
from deephyper.search.hps import CBO, AMBS
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from py_script.dataset import GMD, MultiGMD
from py_script.model import HGT
from py_script.transforms import NormalizeColumnFeatures

import matplotlib.pyplot as plt

import pandas as pd

torch.manual_seed(12345)

# %% [run]

"""TODO:
* fix the issue of the `run` function, return accuracy of validation set
* update the GMDs with new class init
* update the model with new class init
* clean up the code
* test with multiple objectives, return accuracy and ROC-AUC scoore the validation set
"""


def run(config):
    # NOTE: the run function is the function to __maximize__.

    pre_transform = T.Compose([NormalizeColumnFeatures(["x", "edge_attr"])])
    pre_transform = None

    setting = "gic"
    weight_arg = True
    # dataset = GMD(ROOT,
    #               name="uiuc150",
    #               setting=setting,
    #               force_reprocess=False,
    #               pre_transform=pre_transform)
    dataset = MultiGMD(ROOT,
                       names=["epri21", "uiuc150"],
                       setting=setting,
                       force_reprocess=False,
                       pre_transform=pre_transform)
    data = dataset[0]

    batch_size = config.get("batch_size", 64)
    conv_type = config.get("conv_type", "hgt")
    hidden_channels = config.get("hidden_channels", 64)
    num_mlp_layers = config.get("num_mlp_layers", 64)
    activation = config.get("activation", "relu")
    num_heads = config.get("num_heads", 2)
    num_layers = config.get("num_layers", 1)
    dropout = config.get("dropout", 0.5)
    lr = config.get("lr", 1e-3)
    weight_decay = config.get("weight_decay", 1e-4)

    # NOTE: validate the hps settings

    epochs = 200

    # Create a DataLoader for our datasets
    loader_train = DataLoader(dataset=dataset_train,
                              batch_size=batch_size,
                              shuffle=True)

    loader_val = DataLoader(dataset=dataset_val,
                            batch_size=batch_size,
                            shuffle=True)

    # loader_test = DataLoader(dataset=dataset_test,
    #                          batch_size=batch_size,
    #                          shuffle=True)

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

    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)

    losses = []
    pbar = tqdm(range(epochs), desc="epri21")
    model.train()
    for epoch in pbar:
        t_loss = 0
        for i, data in enumerate(loader_train):
            data = data.to(DEVICE)
            optimizer.zero_grad()

            # Decide between MLD or GIC
            if setting == "mld":
                out = model(data)[data.load_bus_mask]
                loss = F.mse_loss(out, data['y'])
            else:
                out = model(data, "gmd_bus")
                if weight_arg and (len(data['y'].bincount()) > 1):
                    weight = len(data['y']) / (2 * data['y'].bincount())
                    loss = F.cross_entropy(out, data['y'], weight=weight)
                else:
                    loss = F.cross_entropy(out, data['y'])

                train_acc = (data['y'].detach().cpu().numpy() == out.argmax(
                    dim=1).detach().cpu().numpy()).sum() / len(data['y'])
                # ROC_AUC score can't be computed when there's only one class (all zeroes in the true output)
                if len(data['y'].bincount()) > 1:
                    # There is at least one blocket placed, so we can compute the roc_auc score
                    train_roc_auc = roc_auc_score(data['y'].detach().cpu().numpy(), out.argmax(1).detach().cpu().numpy())
                else:
                    # TODO: change the default value of roc_auc. We're supposed to throw an error, though.
                    train_roc_auc = 0

            loss.backward()
            optimizer.step()

            # FIXED: we don't need to devide by num_graphs
            t_loss += loss.item()

        # Choose how to handle the pbar based on the problem setting
        if setting == "mld":
            pbar.set_postfix({"loss": t_loss})
        # else:
        #     # pbar.set_postfix({"loss": t_loss, "train_acc": train_acc, "roc_auc": roc_auc})
        losses.append(t_loss)

    model.eval()
    val_loss = 0
    for batch in loader_val:
        batch = batch.to(DEVICE)

        # Compute between MLD or GIC
        if setting == "mld":
            pred = model(batch)[batch.load_bus_mask]
            loss = F.mse_loss(pred, batch.y)
        else:
            # Use the mask to prune cases where there are no busses
            pred = model(batch, "gmd_bus")
            test_y = data['y']
            # pred = model(batch, "gmd_bus")[batch.gic_blocker_bus_mask]
            # test_y = data['y'][batch.gic_blocker_bus_mask]

            if weight_arg and (len(batch.y.bincount()) > 1):
                weight = len(batch.y) / (2 * (batch.y).bincount())
                loss = F.cross_entropy(pred, batch.y, weight=weight)
            else:
                loss = F.cross_entropy(pred, batch.y)

            # Some more objective functions
            val_acc = (batch.y.detach().cpu().numpy() == pred.argmax(
                1).detach().cpu().numpy()).sum() / len(batch.y)

            # ROC_AUC score can't be computed when there's only one class (all zeroes in the true output)
            if len(batch.y.bincount()) > 1:
                # There is at least one blocket placed, so we can compute the roc_auc score
                val_roc_auc = roc_auc_score(batch.y.detach().cpu().numpy(), pred.argmax(1).detach().cpu().numpy())
            else:
                # TODO: change the default value of roc_auc. We're supposed to throw an error, though.
                val_roc_auc = 0

        val_loss += loss.item() / batch.num_graphs

    # maximize the value, so return the negative value of loss, or positive acc/roc-auc score.
    # TOFIX: WRONG return value
    # example:
    # https://deephyper.readthedocs.io/en/latest/tutorials/tutorials/colab/HPS_basic_classification_with_tabular_data/notebook.html#Define-the-run-function
    # return (-val_loss, val_acc, roc_auc)
    return (val_acc, val_roc_auc)

    # MOO: (val_acc, val_roc_auc)
    # SOO: -val_loss OR val_acc OR roc_auc

    # return -test_loss
    # return test_acc
    # return roc_auc


# %%
ROOT = osp.join(osp.expanduser("~"), "tmp", "data", "GMD")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device('cpu')

pre_transform = T.Compose([NormalizeColumnFeatures(["x", "edge_attr"])])
pre_transform = None

setting = "gic"
weight_arg = True
dataset = GMD(ROOT,
              name="epri21",
              setting=setting,
              force_reprocess=False,
              pre_transform=pre_transform)
# dataset = MultiGMD(ROOT,
#                    names=["epri21", "uiuc150"],
#                    setting=setting,
#                    force_reprocess=True,
#                    pre_transform=pre_transform)
data = dataset[0]

# %%
# train/val/test: 0.8/0.1/0.1
dataset_train, dataset_test = train_test_split(dataset, test_size=0.2, random_state=41)
dataset_val, dataset_test = train_test_split(dataset_test, test_size=0.5, random_state=41)

import torch
base_cls = torch.nn.Module
base_cls_repr = 'Act'
acts = [
    act for act in vars(torch.nn.modules.activation).values()
    if isinstance(act, type) and issubclass(act, base_cls)
]

problem = HpProblem()
# TODO: add hyperparameters
# Note: hidden_channels, hidden_size, and batch_size orignally didn't have [1, 2, 4, 8].
#       I added those only because DeepHyper will error out if 1 isn't one of the
#       hyperparameter options.
# problem.add_hyperparameter([16, 32, 64, 128, 256],
#                            "hidden_size", default_value=128)
# problem.add_hyperparameter([16, 32, 64, 128, 256],
#                            "batch_size", default_value=64)

# DEBUG: must include hidden_size=1
problem.add_hyperparameter([1, 16, 32, 64, 128, 256],
                           "hidden_size", default_value=128)
problem.add_hyperparameter([1, 16, 32, 64, 128, 256],
                           "batch_size", default_value=64)
problem.add_hyperparameter([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           "num_conv_layers", default_value=1)
problem.add_hyperparameter([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           "num_mlp_layers", default_value=1)
activation_choices = ["relu", "sigmoid", "tanh", "elu", "leakyrelu"]
problem.add_hyperparameter(activation_choices, "activation", default_value="relu")
problem.add_hyperparameter(["hgt", "han"],
                           "conv_type", default_value="hgt")
problem.add_hyperparameter([1, 2, 4, 8],
                           "num_heads", default_value=2)
problem.add_hyperparameter([0., 0.1, 0.2, 0.3, 0.4, 0.5],
                           "dropout", default_value=0.5)
problem.add_hyperparameter((1e-5, 1e-1, "log-uniform"),
                           "lr", default_value=1e-3)
problem.add_hyperparameter([0., 1e-5, 5e-5, 1e-4],
                           "weight_decay", default_value=0.)

evaluator = Evaluator.create(run,
                             method="thread",
                              method_kwargs={
                             #      "address": None,
                             #      "num_gpus": 1,
                             #      "num_gpus_per_task": 1,
                             #      "num_cpus": 32,
                             #      "num_cpus_per_task": 4,
                             "num_workers": 32, 
                                  "callbacks": [TqdmCallback()]
                              }
                             )
print("Number of workers: ", evaluator.num_workers)

# %%
search = CBO(problem, evaluator, initial_points=[problem.default_configuration])
# Print all the results
print("All results:")
results = search.search(max_evals=200)
print(results)

# 1. get the best HPS setting from validation set
# 2. train the model again with best HPS setting
# 3. report the test metrics based on the best HPS

# Print the best result
# best_objective_index = results[:]['objective'].argmin()
# print("Best results:")
# print(results.iloc[best_objective_index][0:-3])  # The last 3 slots don't matter

best_objective_index = results[:]['objective_0'].argmax()
print("Best result for -test_loss:")
print(results.iloc[best_objective_index][0:-3], '\n')

best_objective_index = results[:]['objective_1'].argmax()
print("Best results for test_acc:")
print(results.iloc[best_objective_index][0:-3], '\n')

best_objective_index = results[:]['objective_2'].argmax()
print("Best results for roc_auc score:")
print(results.iloc[best_objective_index][0:-3], '\n')


# Radar (Spider) plot of the 3 objectives
# Store the labels for the radar plot
labels = ["-test_loss", "test_acc", "roc_auc"]

# Number of variables we're plotting.
num_vars = len(labels)

# Spread out the labels evenly
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# Since the plot is a circle, finish the loop
angles += angles[:1]

# Plot the figure
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# Helper function to plot each row of the results.


def add_row_to_plot(row, color):
    values = results.loc[row, ["objective_0", "objective_1", "objective_2"]].tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=1, label=row)
    ax.fill(angles, values, color=color, alpha=0.25)


# Add each row to the plot
for row, _ in results.iterrows():
    add_row_to_plot(row, "blue")

plt.savefig("HPS Test.png")

# Fix axis to go in the right order and start at 12 o'clock.
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Draw axis lines for each angle and label.
ax.set_thetagrids(np.degrees(angles[:-1]), labels)

# Go through labels and adjust alignment based on where
# it is in the circle.
for label, angle in zip(ax.get_xticklabels(), angles):
    if angle in (0, np.pi):
        label.set_horizontalalignment('center')
    elif 0 < angle < np.pi:
        label.set_horizontalalignment('left')
    else:
        label.set_horizontalalignment('right')

# https://plotly.com/python/radar-chart/
# https://www.pythoncharts.com/matplotlib/radar-charts/

# %%
# python demo_train.py --force --names epri21 --setting gic
# --activation relu --batch_size 64 --conv_type hgt --dropout 0.5
# --hidden_size 128 --lr 5e-4 --num_conv_layers 1 --num_heads 2
# --num_mlp_layers 1 --weight_decay 1e-4 --epochs 250 --weight
