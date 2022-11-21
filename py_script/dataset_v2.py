"""
Create a dataset with HeteroData format.
"""
import os
import os.path as osp
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData, InMemoryDataset

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
                 name: Optional[str] = "b4gic",
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
        h_data = HeteroData()

        # use the normal bus features
        h_data['bus'].x = torch.tensor(mpc['bus'].iloc[:, 1:].to_numpy(), dtype=torch.float32)

        # concatenate bus and gen together
        bus_gen = pd.merge(mpc['bus'], mpc['gen'], how="right", left_on="bus_i", right_on="bus")
        bus_gen = pd.concat([pd.DataFrame(np.eye(4)[bus_gen.type.to_numpy(dtype=int)]).add_prefix("t"), bus_gen],
                            axis=1)
        bus_gen = bus_gen.drop(['bus_i', 'bus', 'type'], axis=1)
        # h_data['gen'].x = torch.tensor(bus_gen.to_numpy(), dtype=torch.float32)

        # process edge_index
        edges = mpc['branch'].iloc[:, :2].to_numpy()
        h_data['bus', 'branch', 'bus'].edge_index = torch.tensor(edges.T - 1, dtype=torch.long)
        h_data['bus', 'branch', 'bus'].edge_attr = torch.tensor(
            mpc['branch'].iloc[:, 2:].to_numpy(), dtype=torch.float32)

        # add a dummy y
        h_data['y'] = h_data['bus'].x[:, 11]

        ''' TODO: DC network with GMD data'''
        # x = torch.tensor(mpc['bus'].iloc[:, 1:].to_numpy(), dtype=torch.float32)

        # # shift the index
        # edges = mpc['branch'].iloc[:, :2].to_numpy() - 1
        # edge_index = torch.tensor(edges).T

        # edge_attr = mpc['branch'].iloc[:, 2:].to_numpy()
        # edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

        # pos = mpc['bus_gmd'].to_numpy()
        # pos = torch.tensor(pos, dtype=torch.float32)

        # # REVIEW: data label or not (y)?
        # # data = Data(x=x, edge_index=edge_index, edge_attr=None,)
        # # TODO: hetero_graph:
        # # https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.HeteroData
        # h_data = HeteroData()

        # ''' process nodes '''
        # # process `bus`
        # h_data['bus'].num_nodes = mpc['bus']['bus_i'].max()
        # mpc['bus'] = pd.concat([pd.get_dummies(mpc['bus'].type), mpc['bus']], axis=1)
        # mpc['bus'] = mpc['bus'].drop(["type"], axis=1)
        # h_data['bus'].x = mpc['bus'].iloc[:, 1:].to_numpy()

        # # process `gen`
        # # mpc['gen'] = mpc['gen'].drop(['bus_i'], axis=1)
        # h_data['gen'].x = mpc['gen'].iloc[:, 1:].to_numpy()

        # # process `gmd_bus`
        # mpc['gmd_bus_ac'] = mpc['gmd_bus'][mpc['gmd_bus']['name'].str.contains("sub")]
        # mpc['gmd_bus_dc'] = mpc['gmd_bus'][mpc['gmd_bus']['name'].str.contains("bus")]
        # h_data['gmd_bus_ac'].x = mpc['gmd_bus_ac'].iloc[:, 1:3].to_numpy()
        # h_data['gmd_bus_dc'].x = mpc['gmd_bus_dc'].iloc[:, 1:3].to_numpy()

        # ''' process edges '''
        # mpc['branch_gmd'] = pd.concat([pd.get_dummies(mpc['branch_gmd'].type), mpc['branch_gmd']], axis=1)
        # mpc['branch_gmd'] = pd.concat([pd.get_dummies(mpc['branch_gmd'].config), mpc['branch_gmd']], axis=1)
        # mpc['branch_gmd'] = mpc['branch_gmd'].drop(['type', 'config'], axis=1)

        # dc_edge_index = mpc['branch'].iloc[:, :2].to_numpy()
        # ac_edge_index = mpc['gmd_branch'].iloc[:, :2].to_numpy()
        # dc_edge_attr = np.concatenate([mpc['branch'].iloc[:, 2:].to_numpy(),
        #                                mpc['branch_gmd'].iloc[:, 4:].to_numpy()], axis=1)
        # ac_edge_attr = mpc['gmd_branch'].iloc[:, 2:].to_numpy()

        # # TODO: verify the type of edges
        # # for the multi-edge settings
        # h_data['bus', 'branch_t1', 'bus'].edge_index = mpc['branch'].iloc[:, :2].to_numpy()
        # h_data['bus', 'branch_t2', 'bus'].edge_index = mpc['branch'].iloc[mpc['branch'].unique(axis=1), :2].to_numpy()
        # h_data['bus', 'branch', 'gen'].edge_index = mpc['branch'].iloc[mpc['gen'].iloc[:, 1:], :2].to_numpy()
        # h_data['bus', 'branch_gmd', 'bus'].edge_index = mpc['branch_gmd'].iloc[:, :2].to_numpy()
        # # TODO: to verify the link of edges
        # h_data['bus', 'attach', 'gmd_bus_ac'].edge_index = mpc['gmd_bus_ac'].iloc[:, :2].to_numpy()
        # h_data['bus', 'attach', 'gmd_bus_dc'].edge_index = mpc['gmd_bus_dc'].iloc[:, :2].to_numpy()
        # # h_data['branch', 'branch'].edge_attr = dc_edge_attr
        # # h_data['branch', 'gmd_branch'].edge_attr = ac_edge_attr

        # # TODO: verify the edge features - more than 2 edges
        # h_data['bus', 'branch_t1', 'bus'].edge_attr = mpc['branch'].iloc[:, 2:].to_numpy()
        # h_data['bus', 'branch_t2', 'bus'].edge_attr = mpc['branch'].iloc[mpc['branch'].unique(axis=1), 2:].to_numpy()
        # h_data['bus', 'branch', 'gen'].edge_attr = mpc['branch'].iloc[mpc['gen'].iloc[:, 1:], 2:].to_numpy()
        # h_data['bus', 'branch_gmd', 'bus'].edge_attr = mpc['branch_gmd'].iloc[:, 2:].to_numpy()
        # h_data['bus', 'attach', 'gmd_bus_ac'].edge_attr = mpc['gmd_bus_ac'].iloc[:, 2:].to_numpy()
        # h_data['bus', 'attach', 'gmd_bus_dc'].edge_attr = mpc['gmd_bus_dc'].iloc[:, 2:].to_numpy()

        # h_data['bus'].y = JuMP
        # save to the processed path
        torch.save(self.collate([h_data]), self.processed_paths[0])

    def __repr__(self) -> str:
        arg_repr = str(len(self)) if len(self) > 1 else ''
        return f'{self.__class__.__name__}({arg_repr}) {self.name}'