"""
Create a dataset with HeteroData format.
"""
import json
import os.path as osp
import shutil
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData, InMemoryDataset

from py_script.utils import create_dir, read_file


class GMD(InMemoryDataset):
    def __init__(self, root: Optional[str] = None,
                 name: Optional[str] = "b4gic",
                 transform: Optional[Callable] = None,
                 force_reprocess: Optional[bool] = False,
                 problem: Optional[str] = 'clf',
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):

        self.root = root
        self.name = name
        self.solution_name = name + "_blocker_placement_results"
        self.transform = transform
        self.force_reprocess = force_reprocess
        self.problem = problem
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        if self.force_reprocess:
            dir_path = osp.dirname(osp.realpath(__file__))
            SAVED_PATH = osp.join(dir_path, "processed", self.name)
            if osp.exists(SAVED_PATH):
                shutil.rmtree(SAVED_PATH)
                shutil.rmtree(osp.join(self.root, "processed"))

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
        # Input filename
        fn = self.root + "/" + self.name + ".m"
        mpc = read_file(fn)

        # Soultion filename
        fn = self.root + "/results/" + self.name + "_results.json"
        dc_placement = json.load(open(fn))
        res_gmd_bus = pd.DataFrame.from_dict(dc_placement['solution']['gmd_bus']).T.sort_index()
        res_gmd_bus = res_gmd_bus.drop(['source_id'], axis=1)

        h_data = HeteroData()

        ''' node_type: bus '''
        # convert the `type` to one-hot encoder
        mpc['bus'] = pd.concat([mpc['bus'], pd.DataFrame(
            np.eye(4)[mpc['bus'].type.to_numpy(dtype=int)]).add_prefix("t")], axis=1)
        h_data['bus'].x = torch.tensor(mpc['bus'].iloc[:, 1:].to_numpy(), dtype=torch.float32)

        # build the bus_i to node_i mapping
        n_nodes = mpc['bus'].shape[0]
        mapping = {}
        for i in range(n_nodes):
            mapping[mpc['bus'].bus_i[i]] = i

        ''' node_type: gen '''
        # creating new virtual link between bus and gen to handle multiple generators
        h_data['gen'].x = torch.tensor(mpc['gen'].iloc[:, 1:].to_numpy(), dtype=torch.float32)

        ''' edge_type (virtual): gen--conn--bus '''
        n_gen = mpc['gen'].shape[0]
        gen_bus_edges = np.zeros((n_gen, 2))
        for i in range(n_gen):
            gen_bus_edges[i] = [i, mapping[mpc['gen'].bus[i]]]
        # edge feature
        h_data['gen', 'conn', "bus"].edge_index = torch.tensor(gen_bus_edges.T, dtype=torch.long)

        # convert the tuples with mapping
        ''' edge_type: bus--branch--bus '''
        n_branch = mpc['branch'].shape[0]
        bus_bus_edges = np.zeros((n_branch, 2))
        for i in range(n_branch):
            bus_bus_edges[i] = [mapping[mpc['branch'].fbus[i]], mapping[mpc['branch'].tbus[i]]]
        h_data['bus', 'branch', 'bus'].edge_index = torch.tensor(bus_bus_edges.T, dtype=torch.long)
        h_data['bus', 'branch', 'bus'].edge_attr = torch.tensor(
            mpc['branch'].iloc[:, 2:].to_numpy(), dtype=torch.float32)

        ''' edge_type: bus--branch_gmd--bus '''
        # convert type and config to one-hot encoder
        mpc['branch_gmd'] = pd.concat([mpc['branch_gmd'], pd.get_dummies(mpc['branch_gmd'].type)], axis=1)
        mpc['branch_gmd'] = pd.concat([mpc['branch_gmd'], pd.get_dummies(mpc['branch_gmd'].config)], axis=1)
        mpc['branch_gmd'] = mpc['branch_gmd'].drop(['type', 'config'], axis=1)

        n_branch_gmd = mpc['branch_gmd'].shape[0]
        edges = np.zeros((n_branch_gmd, 2))
        for i in range(n_branch_gmd):
            edges[i] = [mapping[mpc['branch_gmd'].hi_bus[i]], mapping[mpc['branch_gmd'].lo_bus[i]]]
        h_data['bus', 'branch_gmd', 'bus'].edge_index = torch.tensor(edges.T, dtype=torch.long)
        h_data['bus', 'branch_gmd', 'bus'].edge_attr = torch.tensor(
            mpc['branch_gmd'].iloc[:, 2:].to_numpy(), dtype=torch.float32)

        pos = mpc['bus_gmd'].to_numpy()
        pos = torch.tensor(pos, dtype=torch.float32)

        ''' DC network with GMD data '''
        # process `gmd_bus` from PowerModelsGMD results
        if self.problem == 'clf':
            # classification problem
            gmd_bus_attr = res_gmd_bus[['gmd_vdc', 'status', 'g_gnd']].astype('float').to_numpy()
            h_data['gmd_bus'].x = torch.tensor(gmd_bus_attr, dtype=torch.float32)

            # same dimension as `gmd_bus`
            h_data['y'] = torch.tensor(res_gmd_bus['blocker_placed'].astype("int").to_numpy(), dtype=torch.long)

        elif self.problem == "reg":
            # regression problem
            gmd_bus_attr = res_gmd_bus[['blocker_placed', 'status', 'g_gnd']].astype('float').to_numpy()
            h_data['gmd_bus'].x = torch.tensor(gmd_bus_attr, dtype=torch.float32)

            # same dimension as `gmd_bus`
            y = res_gmd_bus['gmd_vdc'].astype("float").to_numpy()
            # REVIEW: normalize
            # y = (y-y.min()) / (y.max() - y.min())
            h_data['y'] = torch.tensor(y, dtype=torch.float32)
        else:
            raise Exception("Unknown problem setting, `clf` or `reg` only.")

        ''' edge_type: gmd_bus--gmd_branch--gmd_bus '''
        gmd_edges = mpc['gmd_branch'].iloc[:, :2].to_numpy()
        # gmd edge index
        h_data['gmd_bus', 'gmd_branch', 'gmd_bus'].edge_index = torch.tensor(gmd_edges.T - 1, dtype=torch.long)
        h_data['gmd_bus', 'gmd_branch', 'gmd_bus'].edge_attr = torch.tensor(
            mpc['gmd_branch'].iloc[:, 3:-1].to_numpy(), dtype=torch.float32)

        ''' node_type: gmd_bus '''
        n_gmd_bus = mpc['gmd_bus'].shape[0]
        gmd_bus_bus_edges = np.zeros((n_gmd_bus, 2))
        for i in range(n_gmd_bus):
            gmd_bus_bus_edges[i] = [i, mapping[mpc['gmd_bus'].parent_index[i]]]

        ''' edge_type (virtual): gmd_bus--attach--bus '''
        h_data['gmd_bus', 'attach', "bus"].edge_index = torch.tensor(gmd_bus_bus_edges.T, dtype=torch.long)

        # save to the processed path
        torch.save(self.collate([h_data]), self.processed_paths[0])

    def __repr__(self) -> str:
        arg_repr = str(len(self)) if len(self) > 1 else ''
        return f'{self.__class__.__name__}({arg_repr}) {self.name}'
