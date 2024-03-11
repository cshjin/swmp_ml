""" Demo training script for HGT on GMD datasets
"""

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
from py_script.model import HeteroGNN
from py_script.transforms import NormalizeColumnFeatures
from py_script.utils import create_dir, get_device, process_args

if __name__ == "__main__":
    args = process_args()

    if args['no_norm']:
        pre_transform = None
    else:
        pre_transform = T.Compose([NormalizeColumnFeatures(['x', 'edge_attr'])])

    if args['seed'] != -1:
        SEED = args['seed']
        torch.manual_seed(SEED)
        np.random.seed(SEED)
    else:
        SEED = np.random.randint(0, 10000)

    # ROOT folder for the GMD datasets to be stored
    ROOT = osp.join('/tmp', 'data', 'GMD')
    if args['log']:
        DT_FORMAT = datetime.now().strftime("%Y%m%d_%H%M%S")
        LOG_DIR = osp.join('logs', 'demo_train', DT_FORMAT)
        create_dir(LOG_DIR)

    # Select the processor to use
    DEVICE = get_device(args['gpu'])

    # Create the dataset
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

    # sample data
    data = dataset[0]

    # Train and test split for our datasets with ratio: 0.8/0.1/0.1
    dataset_train, dataset_test = train_test_split(dataset,
                                                   test_size=0.2,
                                                   random_state=SEED,
                                                   shuffle=True)
    dataset_val, dataset_test = train_test_split(dataset_test,
                                                 test_size=0.5,
                                                 random_state=SEED,
                                                 shuffle=True)

    # Create a DataLoader for our datasets
    # NOTE: update the batch size to `1` for reevaluation
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
    model = HeteroGNN(hidden_channels=args['hidden_size'],
                      num_mlp_layers=args['num_mlp_layers'],
                      conv_type=args['conv_type'],
                      act=args['act'],
                      out_channels=out_channels,
                      num_conv_layers=args['num_conv_layers'],
                      num_heads=args['num_heads'],
                      dropout=args['dropout'],
                      metadata=data.metadata(),
                      device=DEVICE,
                      ).to(DEVICE)

    model.load_state_dict(torch.load("_model.pt"))
    model.eval()

    res = model.evaluate(data_loader_test, device=DEVICE)
    print(f"Test acc: {res[0]:.4f}, roc-auc: {res[1]:.4f}, loss: {res[2]:.4f}")
