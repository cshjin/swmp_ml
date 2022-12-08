"""Test loading the DC blocker JSON file"""

import os
import os.path as osp
from typing import Callable, Optional

import json

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData, InMemoryDataset

from py_script.utils import read_file

# Open the JSON file
directory = "./test/data/"
file_name = "b4gic_blocker_placement_results.json"
path = directory + file_name
dc_blocker_file = open(path)

# Load the data as a dictionary
dc_placement = json.load(dc_blocker_file)

# print(dc_placement["input"]["gmd_bus"]["blocker_placed"])

solution = {}
for i in dc_placement["input"]["gmd_bus"]:
    solution[i] = dc_placement["input"]["gmd_bus"][i]["blocker_placed"]

print(solution)
# ["input"]["gmd_bus"]
# ["blocker_placed"]
# print(dc_placement.values())