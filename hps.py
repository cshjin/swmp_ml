# %%
import os.path as osp
from datetime import datetime

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
from py_script.model import HeteroGNN
from py_script.transforms import NormalizeColumnFeatures
from py_script.utils import create_dir

torch.manual_seed(12345)


# %%
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

    # Create a DataLoader for our datasets
    loader_train = DataLoader(dataset=dataset_train,
                              batch_size=batch_size,
                              shuffle=True)

    loader_val = DataLoader(dataset=dataset_val,
                            batch_size=batch_size,
                            shuffle=True)

    model = HeteroGNN(hidden_channels=hidden_channels,
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

    # Total number of epochs
    epochs = 200

    # early stopping
    best_val_metrics = [0, 0, 1e4]
    patience, count = 5, 0
    for epoch in range(epochs):
        # Train the model for one epoch
        _, _, _ = model.fit(optimizer, loader_train, epochs=1, verbose=False)

        # Evaluate the model on the validation set
        val_acc, val_roc_auc, val_loss = model.evaluate(loader_val)

        if val_loss < best_val_metrics[2]:
            best_val_metrics = [val_acc, val_roc_auc, val_loss]
            count = 0
        else:
            count += 1

        # Stop early if we hit the threshold
        if count >= patience:
            return best_val_metrics[0]

    return best_val_metrics[0]


# %%
ROOT = osp.join("/tmp", "data", "GMD")
DT_FORMAT = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = osp.join("logs", "hps", DT_FORMAT)
create_dir(LOG_DIR)

# REVIEW: why we have to use CPU?
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_gpu_available = torch.cuda.is_available()
n_gpus = torch.cuda.device_count()
DEVICE = torch.device('cuda' if is_gpu_available else 'cpu')
# DEVICE = torch.device('cpu')

pre_transform = T.Compose([NormalizeColumnFeatures(["x", "edge_attr"])])

setting = "gic"
weight_arg = True
dataset = GMD(ROOT,
              name="epri21",
              setting=setting,
              force_reprocess=True,
              pre_transform=pre_transform)

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

# add kwargs for different methods
NUM_WORKERS = 8
serial_kwargs = {"num_workers": NUM_WORKERS,
                 "callbacks": [TqdmCallback()]}

thread_kwargs = {"num_workers": NUM_WORKERS,
                 "callbacks": [TqdmCallback()]}

process_kwargs = {"num_workers": NUM_WORKERS,
                  "callbacks": [TqdmCallback()]}

ray_kwargs = {
    "num_cpus": 1,
    "num_cpus_per_task": 1,
    "callbacks": [TqdmCallback()]
}

if is_gpu_available:
    ray_kwargs["num_cpus"] = n_gpus
    ray_kwargs["num_cpus_per_task"] = 1
    ray_kwargs["num_gpus"] = n_gpus
    ray_kwargs["num_gpus_per_task"] = 1

ray_kwargs = {"num_cpus": 4,
              "num_gpus": 1,
              "num_cpus_per_task": 1,
              "num_gpus_per_task": 1,
              "callbacks": [TqdmCallback()]}

evaluator = Evaluator.create(run,
                             method="serial",
                             method_kwargs=serial_kwargs,
                             )

print("Number of workers: ", evaluator.num_workers)

search = CBO(problem, evaluator, initial_points=[problem.default_configuration], random_state=42)

# Print all the results
results = search.search(max_evals=200, timeout=3600)
# print(results)
results.to_csv(osp.join(LOG_DIR, "hps_results.csv"))

# %%
''' Find the best HPS setting based on the best objective (SOO) '''

results.to_csv(osp.join(LOG_DIR, "hps_results.csv"))
idx_candidate = results.loc[results['objective'] == results['objective'].max()].index

best_model = None
best_optimizer = None
hp_idx = None
lowest_parameter_count = 1e10

for idx in idx_candidate:
    hp_idx = results.iloc[idx]
    _model = HeteroGNN(hidden_channels=hp_idx['p:hidden_size'],
                       conv_type=hp_idx['p:conv_type'],
                       num_mlp_layers=hp_idx['p:num_mlp_layers'],
                       activation=hp_idx['p:activation'],
                       out_channels=2,
                       num_heads=hp_idx['p:num_heads'],
                       num_conv_layers=hp_idx['p:num_conv_layers'],
                       dropout=hp_idx['p:dropout'],
                       node_types=data.node_types,
                       metadata=data.metadata(),
                       ).to(DEVICE)
    # optimizer = torch.optim.Adam(_model.parameters(), lr=hp_idx['p:lr'], weight_decay=hp_idx['p:weight_decay'])
    _model(dataset_train[0])
    total_paramenter_count = sum(parameter.numel() for parameter in _model.parameters() if parameter.requires_grad)

    if total_paramenter_count < lowest_parameter_count:
        best_idx = idx
        lowest_parameter_count = total_paramenter_count
        # best_model = _model
        # best_optimizer = optimizer
        # hp_idx = results.iloc[idx][0:-3].to_dict()
print(f"best idx {best_idx}, num param {lowest_parameter_count}")

# %%
''' retrain based on best hp to get best model '''
best_hp = results.iloc[best_idx]

loader_train = DataLoader(dataset=dataset_train,
                          batch_size=int(hp_idx['p:batch_size']),
                          shuffle=True)

loader_val = DataLoader(dataset=dataset_val,
                        batch_size=int(hp_idx['p:batch_size']),
                        shuffle=True)

loader_test = DataLoader(dataset=dataset_test,
                         batch_size=int(hp_idx['p:batch_size']),
                         shuffle=True)


# retrain the model with best HPS setting
best_model = HeteroGNN(hidden_channels=best_hp['p:hidden_size'],
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
current_optimizer = torch.optim.Adam(
    best_model.parameters(),
    lr=best_hp['p:lr'],
    weight_decay=best_hp['p:weight_decay'])

best_val_loss = 1e10
for epoch in range(200):
    train_metrics = best_model.fit(current_optimizer, loader_train, epochs=1, verbose=False)
    val_acc, val_roc, val_loss = best_model.evaluate(loader_val)
    # print(f"Epoch {epoch + 1:03d}, Val Loss: {val_loss:.4f}")
    # Save best accuracy
    if val_loss > best_val_loss:
        best_val_loss = val_loss
        torch.save(best_model.state_dict(), osp.join(LOG_DIR, "best_model.pt"))

# Load our best current model and evaluate test accuracy
if osp.exists(osp.join(LOG_DIR, "best_model.pt")):
    best_model.load_state_dict(torch.load(osp.join(LOG_DIR, "best_model.pt")))
test_acc, test_roc_auc, test_loss = best_model.evaluate(loader_test)

print("best hps", best_hp,
      "test acc", test_acc, "\n",
      "test roc auc", test_roc_auc)
