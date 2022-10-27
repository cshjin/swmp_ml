import os
import os.path as osp
from typing import Callable, Optional

import numpy as np
import torch
from pandapower.converter import from_mpc
from pandapower.topology import create_nxgraph
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

        # TODO: process the file either using pandapower or from raw data file.

        # filename
        fn = self.root + "/" + self.name + ".m"
        # NOTE: Deprecated. Switch from pandapower to raw data processing.
        #       pandapower is not accurate
        # net = from_mpc(fn)
        # g = create_nxgraph(net)
        # x = torch.tensor(net.bus[['name', 'vn_kv', 'zone', 'in_service', 'min_vm_pu',
        #                  'max_vm_pu']].astype(float).to_numpy(), dtype=torch.float32)

        # edges = np.array(list(g.edges()))
        # edge_index = torch.tensor(edges).T
        # print(f"# of nodes {len(g.nodes())}",
        #       f"# of edges {len(g.edges())}")

        # NOTE: process from raw data
        mpc = read_file(fn)
        x = torch.tensor(mpc['bus'].iloc[:, 1:].to_numpy(), dtype=torch.float32)

        # shift the index
        edges = mpc['branch'].iloc[:, :2].to_numpy() - 1
        edge_index = torch.tensor(edges).T

        edge_attr = mpc['branch'].iloc[:, 2:].to_numpy()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

        pos = mpc['bus_gmd'].to_numpy()
        pos = torch.tensor(pos, dtype=torch.float32)

        # REVIEW: data label or not (y)?
        data = Data(x=x, edge_index=edge_index)
        # save to the processed path
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        arg_repr = str(len(self)) if len(self) > 1 else ''
        return f'{self.__class__.__name__}({arg_repr}) {self.name}'
