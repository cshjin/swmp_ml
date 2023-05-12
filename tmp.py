# %%  load python directly

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