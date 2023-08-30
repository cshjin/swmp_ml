# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os.path as osp
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from py_script.dataset import GMD, MultiGMD
from py_script.model import HPIGNN
from py_script.transforms import NormalizeColumnFeatures
from py_script.utils import (create_dir, export_gmd_blocker_matpower,
                             get_device, process_args)

SEED = 12345
torch.manual_seed(SEED)
np.random.seed(SEED)

# %%
pre_transform = T.Compose([NormalizeColumnFeatures(['x', 'edge_attr'])])
ROOT = osp.join('/tmp', 'data', 'GMD')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = GMD(ROOT,
              name="epri21",
              setting="gic",
              force_reprocess=False,
              pre_transform=pre_transform)
data = dataset[0]

# Train and test split for our datasets
dataset_train, dataset_test = train_test_split(dataset,
                                               test_size=0.2,
                                               random_state=SEED,
                                               shuffle=True)
dataset_train, dataset_val = train_test_split(dataset_test,
                                              test_size=0.5,
                                              random_state=SEED,
                                              shuffle=True)
data_loader_train = DataLoader(dataset=dataset_train,
                               batch_size=1,
                               shuffle=True)
data_loader_val = DataLoader(dataset=dataset_val,
                             batch_size=1,
                             shuffle=True)
data_loader_test = DataLoader(dataset=dataset_test,
                              batch_size=1,
                              shuffle=True)
# %%

''' export the gmd_blocker labels from heuristic approach '''


# for test_idx in range(len(dataset_test)):
#     export_gmd_blocker_matpower(dataset_test[test_idx].y.numpy(),
#                                 in_fn="test/data/epri21.m",
#                                 out_fn=f"epri21_true_{test_idx:02d}.m")

# %%

''' train the GNN model to predict the gic blockers'''
args = {"hidden_size": 64,
        "num_mlp_layers": 6,
        "num_conv_layers": 8,
        "num_heads": 2,
        "dropout": 0.5,
        "conv_type": "han",
        "activation": "relu",
        "batch_size": 128,
        "lr": 1e-3,
        "epochs": 200,
        "weight_decay": 0.0}
# args = process_args()
model = HPIGNN(hidden_channels=args['hidden_size'],
               num_mlp_layers=args['num_mlp_layers'],
               conv_type=args['conv_type'],
               activation=args['activation'],
               out_channels=2,
               num_conv_layers=args['num_conv_layers'],
               num_heads=args['num_heads'],
               dropout=args['dropout'],
               node_types=data.node_types,
               metadata=data.metadata(),
               device=DEVICE,
               name="epri21",
               ).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(),
                             lr=args['lr'],
                             weight_decay=args['weight_decay'])

res = model.fit(optimizer,
                data_loader_train,
                epochs=args['epochs'],
                eval_freq=10,
                verbose=True,
                device=DEVICE)
# res = model.evaluate(data_loader_test, device=DEVICE)
# print(res)

model.eval_single(dataset_test[0])

# %%

''' export the gmd_blocker labels from GNN approach '''
# for test_idx in range(len(dataset_test)):
#     gic_str = """\n%% gmd_blocker data\n%column_names% gmd_bus status\nmpc.gmd_blocker = {\n"""
#     for idx, v in enumerate(model.eval_single(dataset_test[test_idx])):
#         gic_str += f"\t{idx+1}\t{v.item()}\n"
#     gic_str += "};"

#     with open(f"epri21_pred_{test_idx:02d}.m", "w") as f1:
#         with open("test/data/epri21.m", "r") as f2:
#             f1.write(f2.read() + gic_str)


# for test_idx in range(len(dataset_test)):
#     export_gmd_blocker_matpower(model.eval_single(dataset_test[test_idx]).numpy(),
#                                 in_fn="test/data/epri21.m",
#                                 out_fn=f"epri21_pred_{test_idx:02d}.m")
# %%
