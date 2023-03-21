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
    batch_size = config.get("batch_size", 64)
    hidden_channels = config.get("hidden_channels", 64)
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
                out_channels=1,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
                node_types=data.node_types,
                metadata=data.metadata(),
                ).to(DEVICE)

    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)

    pbar = tqdm(range(200), desc="epri21")
    losses = []
    model.train()
    # for epoch in range(200):
    for epoch in pbar:
        t_loss = 0
        for i, batch in enumerate(loader_train):
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            out = model(batch)
            loss = F.mse_loss(out, batch.y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() / batch.num_graphs
        losses.append(t_loss)

    model.eval()
    test_loss = 0
    for batch in loader_test:
        batch = batch.to(DEVICE)
        pred = model(batch)
        loss = F.mse_loss(pred, batch.y)
        test_loss += loss.item() / batch.num_graphs

    return test_loss


# %%
ROOT = osp.join(osp.expanduser("~"), "tmp", "data", "GMD")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device('cpu')

pre_transform = T.Compose([NormalizeColumnFeatures(["x", "edge_attr"])])

dataset = GMD(ROOT,
              name="epri21",
              problem="reg",
              force_reprocess=False,
              pre_transform=pre_transform)
data = dataset[0]

# %%
dataset_train, dataset_test = train_test_split(dataset, test_size=0.2, random_state=41)

problem = HpProblem()
# TODO: add hyperparameters
problem.add_hyperparameter([16, 32, 64, 128, 256],
                           "hidden_channels", default_value=128)
problem.add_hyperparameter([16, 32, 64, 128, 256],
                           "batch_size", default_value=64)
problem.add_hyperparameter([1, 2, 3, 4, 5, 6],
                           "num_layers", default_value=1)
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
