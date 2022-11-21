import os
import os.path as osp
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

from py_script.utils import read_file


def create_dir(path):
    """ Create a dir where the processed data will be stored
    Args:
        path (str): Path to create the folder.
    """
    dir_exists = os.path.exists(path)

    if not dir_exists:
        try:
            os.makedirs(path)
            print("The {} directory is created.".format(path))
        except Exception as e:
            print("Error: {}".format(e))
            exit(-1)


class GMD(InMemoryDataset):
    def __init__(self, root: Optional[str] = None,
                 name: Optional[str] = "epri21",
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):

        self.root = root
        self.name = name
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        dir_path = osp.dirname(osp.realpath(__file__))
        SAVED_PATH = osp.join(dir_path, "processed", self.name)
        create_dir(SAVED_PATH)
        return [f'{SAVED_PATH}/processed.pt']

    def process(self):
        """ Process the raw file, and save to processed files. """
        # filename
        fn = self.root + "/" + self.name + ".m"
        mpc = read_file(fn)

        bus_gen = pd.merge(mpc['bus'], mpc['gen'], how="left", left_on="bus_i", right_on="bus")
        # bus has 4 types
        bus_gen = pd.concat([pd.DataFrame(np.eye(4)[bus_gen.type.to_numpy(dtype=int)]).add_prefix("t"), bus_gen],
                            axis=1)
        bus_gen = bus_gen.drop(['bus_i', 'bus', 'type'], axis=1)
        x = torch.tensor(np.nan_to_num(bus_gen.to_numpy(), nan=-1), dtype=torch.float32)
        # process edge index and features
        edges = mpc['branch'].iloc[:, :2].to_numpy()
        edge_index = torch.tensor(edges.T - 1, dtype=torch.long)
        edge_attr = torch.tensor(mpc['branch'].iloc[:, 2:].to_numpy(), dtype=torch.float32)

        # NOTE: dummy y
        data = Data(x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=x[:, 11])

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        arg_repr = str(len(self)) if len(self) > 1 else ''
        return f'{self.__class__.__name__}({arg_repr}) {self.name}'
