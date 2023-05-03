# %%
import os.path as osp

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

from py_script.dataset import GMD
from py_script.model import HGT
from py_script.transforms import NormalizeColumnFeatures

torch.manual_seed(12345)

# %% [run]


def run(config):
    pre_transform = T.Compose([NormalizeColumnFeatures(["x", "edge_attr"])])

    setting="gic"
    weight_arg=False
    dataset = GMD(ROOT,
                    name="epri21",
                    setting=setting,
                    problem="clf",
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

    # Create a DataLoader for our datasets
    loader_train = DataLoader(dataset=dataset_train,
                              batch_size=batch_size,
                              shuffle=True)

    loader_test = DataLoader(dataset=dataset_test,
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

    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)

    # pbar = tqdm(range(200), desc="epri21")
    # losses = []
    # model.train()
    # # for epoch in range(200):
    # for epoch in pbar:
    #     t_loss = 0
    #     for i, batch in enumerate(loader_train):
    #         batch = batch.to(DEVICE)
    #         optimizer.zero_grad()
    #         out = model(batch)
    #         loss = F.mse_loss(out, batch.y)
    #         loss.backward()
    #         optimizer.step()
    #         t_loss += loss.item() / batch.num_graphs
    #     losses.append(t_loss)

    losses = []
    pbar = tqdm(range(200), desc="epri21")
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
                if weight_arg:
                    weight = len(data['y']) / (2 * data['y'].bincount())
                    loss = F.cross_entropy(out, data['y'], weight=weight)
                else:
                    loss = F.cross_entropy(out, data['y'])
                
                train_acc = (data['y'].detach().cpu().numpy() == out.argmax(
                dim=1).detach().cpu().numpy()).sum() / len(data['y'])
                # roc_auc = roc_auc_score(data['y'].detach().cpu().numpy(), out.argmax(1).detach().cpu().numpy())
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
    test_loss = 0
    for batch in loader_test:
        batch = batch.to(DEVICE)

        # Compute between MLD or GIC
        if setting == "mld":
            pred = model(batch)[batch.load_bus_mask]
            loss = F.mse_loss(pred, batch.y)
        else:
            pred = model(batch, "gmd_bus")
            # loss = F.mse_loss(batch, data['y'])
            if weight_arg:
                weight = len(batch.y) / (2 * (batch.y).bincount())
                loss = F.cross_entropy(pred, batch.y, weight=weight)
            else:
                loss = F.cross_entropy(pred, batch.y)
        
        test_loss += loss.item() / batch.num_graphs

    return test_loss


# %%
ROOT = osp.join(osp.expanduser("~"), "tmp", "data", "GMD")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device('cpu')

pre_transform = T.Compose([NormalizeColumnFeatures(["x", "edge_attr"])])

setting="gic"
weight_arg=False
dataset = GMD(ROOT,
                name="epri21",
                setting=setting,
                problem="clf",
                force_reprocess=True,
                pre_transform=pre_transform)
data = dataset[0]

# %%
dataset_train, dataset_test = train_test_split(dataset, test_size=0.2, random_state=41)

import torch
base_cls = torch.nn.Module
base_cls_repr = 'Act'
acts = [
    act for act in vars(torch.nn.modules.activation).values()
    if isinstance(act, type) and issubclass(act, base_cls)
]

problem = HpProblem()
# TODO: add hyperparameters
problem.add_hyperparameter([16, 32, 64, 128, 256],
                           "hidden_channels", default_value=128)
problem.add_hyperparameter([16, 32, 64, 128, 256],
                           "hidden_size", default_value=128)
problem.add_hyperparameter([16, 32, 64, 128, 256],
                           "batch_size", default_value=64)
problem.add_hyperparameter([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           "num_conv_layers", default_value=1)
problem.add_hyperparameter([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           "num_mlp_layers", default_value=1)
activation_choices = ["relu", "sigmoid", "tanh", "elu", "leakyrelu"]
problem.add_hyperparameter(activation_choices, "activation", default_value="relu")
problem.add_hyperparameter([50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800],
                           "epochs", default_value=200)
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
                             #  method_kwargs={
                             #      "address": None,
                             #      "num_gpus": 1,
                             #      "num_gpus_per_task": 1,
                             #      "num_cpus": 32,
                             #      "num_cpus_per_task": 4,
                             #      "callbacks": [TqdmCallback()]
                             #  }
                             )
print("Number of workers: ", evaluator.num_workers)

# %%
search = CBO(problem, evaluator, initial_points=[problem.default_configuration])
results = search.search(10)
print(results)

# %%