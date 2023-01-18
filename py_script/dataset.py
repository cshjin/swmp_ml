"""
Create a dataset with HeteroData format.
"""
import os
import os.path as osp
from typing import Callable, Optional

import json

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData, InMemoryDataset
import shutil
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

        # use the normal bus features
        # convert the `type` to one-hot encoder
        mpc['bus'] = pd.concat([mpc['bus'], pd.DataFrame(
            np.eye(4)[mpc['bus'].type.to_numpy(dtype=int)]).add_prefix("t")], axis=1)
        h_data['bus'].x = torch.tensor(mpc['bus'].iloc[:, 1:].to_numpy(), dtype=torch.float32)

        # creating new virtual link between bus and gen to handle multiple generators
        h_data['gen'].x = torch.tensor(mpc['gen'].iloc[:, 1:].to_numpy(), dtype=torch.float32)
        tmp = mpc['gen'].reset_index()
        tmp['bus'] -= 1
        bus_gen_conn = tmp.iloc[:, :2].to_numpy()
        # edge information
        h_data['gen', 'conn', "bus"].edge_index = torch.tensor(bus_gen_conn.T, dtype=torch.long)

        # DEPRECATED: concatenate bus and gen together
        '''
        bus_gen = pd.merge(mpc['bus'], mpc['gen'], how="right", left_on="bus_i", right_on="bus")
        bus_gen = pd.concat([pd.DataFrame(np.eye(4)[bus_gen.type.to_numpy(dtype=int)]).add_prefix("t"), bus_gen],
                            axis=1)
        bus_gen = bus_gen.drop(['bus_i', 'bus', 'type'], axis=1)
        h_data['gen'].x = torch.tensor(bus_gen.to_numpy(), dtype=torch.float32)
        '''
        # build the mapping
        n_nodes = mpc['bus'].shape[0]
        mapping = {}
        for i in range(n_nodes):
            mapping[mpc['bus'].bus_i[i]] = i

        # convert the tuples with mapping
        n_edges = mpc['branch'].shape[0]
        edges = np.zeros((n_edges, 2))
        for i in range(n_edges):
            edges[i] = [mapping[mpc['branch'].fbus[i]], mapping[mpc['branch'].tbus[i]]]
        # TOFIX: change the edge_index with node_id
        # process edge_index
        # edges = mpc['branch'].iloc[:, :2].to_numpy()
        h_data['bus', 'branch', 'bus'].edge_index = torch.tensor(edges.T, dtype=torch.long)
        h_data['bus', 'branch', 'bus'].edge_attr = torch.tensor(
            mpc['branch'].iloc[:, 2:].to_numpy(), dtype=torch.float32)

        # process `branch_gmd` - AC
        mpc['branch_gmd'] = pd.concat([mpc['branch_gmd'], pd.get_dummies(mpc['branch_gmd'].type)], axis=1)
        mpc['branch_gmd'] = pd.concat([mpc['branch_gmd'], pd.get_dummies(mpc['branch_gmd'].config)], axis=1)
        mpc['branch_gmd'] = mpc['branch_gmd'].drop(['type', 'config'], axis=1)

        # TOFIX: replace edge with new process
        # edges = mpc['branch_gmd'].iloc[:, :2].to_numpy()
        n_branch_gmd = mpc['branch_gmd'].shape[0]
        edges = np.zeros((n_branch_gmd, 2))
        for i in range(n_branch_gmd):
            edges[i] = [mapping[mpc['branch_gmd'].hi_bus[i]], mapping[mpc['branch_gmd'].lo_bus[i]]]
        h_data['bus', 'branch_gmd', 'bus'].edge_index = torch.tensor(edges.T, dtype=torch.long)
        h_data['bus', 'branch_gmd', 'bus'].edge_attr = torch.tensor(
            mpc['branch_gmd'].iloc[:, 2:].to_numpy(), dtype=torch.float32)

        ''' REVIEW: DC network with GMD data'''
        pos = mpc['bus_gmd'].to_numpy()
        pos = torch.tensor(pos, dtype=torch.float32)

        # ''' process nodes '''
        # h_data['gmd_bus'].x = mpc['gmd_bus'].iloc[:, 1:3].to_numpy()
        if self.problem == 'clf':
            # REVIEW: classification problem, read true label from results
            # gmd bus attr
            gmd_bus_attr = res_gmd_bus[['gmd_vdc', 'status', 'g_gnd']].astype('float').to_numpy()
            h_data['gmd_bus'].x = torch.tensor(gmd_bus_attr, dtype=torch.float32)

            # NOTE: same dimension as `gmd_bus`
            h_data['y'] = torch.tensor(res_gmd_bus['blocker_placed'].astype("int").to_numpy(), dtype=torch.long)

        elif self.problem == "reg":
            # REVIEW: regression problem, read true label from results
            # gmd bus attr
            gmd_bus_attr = res_gmd_bus[['blocker_placed', 'status', 'g_gnd']].astype('float').to_numpy()
            h_data['gmd_bus'].x = torch.tensor(gmd_bus_attr, dtype=torch.float32)

            # NOTE: same dimension as `gmd_bus`
            y = res_gmd_bus['gmd_vdc'].astype("float").to_numpy()
            # REVIEW: normalize
            # y = (y-y.min()) / (y.max() - y.min())
            h_data['y'] = torch.tensor(y, dtype=torch.float32)
        else:
            raise Exception("Unknown problem setting, `clf` or `reg` only.")

        gmd_edges = mpc['gmd_branch'].iloc[:, :2].to_numpy()
        # gmd edge index
        h_data['gmd_bus', 'gmd_branch', 'gmd_bus'].edge_index = torch.tensor(gmd_edges.T - 1, dtype=torch.long)
        h_data['gmd_bus', 'gmd_branch', 'gmd_bus'].edge_attr = torch.tensor(mpc['gmd_branch'].iloc[:, 3:-1].to_numpy(),
                                                                            dtype=torch.float32)
        tmp = mpc['gmd_bus'].reset_index()
        # tmp['parent_index'] -= 1
        # ac_dc_attach = tmp.iloc[:, :2].to_numpy()

        # TOFIX: reindex bus
        n_gmd_bus = mpc['gmd_bus'].shape[0]
        ac_dc_attach = np.zeros((n_gmd_bus, 2))
        for i in range(n_gmd_bus):
            ac_dc_attach[i] = [i, mapping[mpc['gmd_bus'].parent_index[i]]]

        # REIVEW: build connection btw AC and DC
        h_data['gmd_bus', 'attach', "bus"].edge_index = torch.tensor(ac_dc_attach.T, dtype=torch.long)

        # DEPRECATED:
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

        # save to the processed path
        torch.save(self.collate([h_data]), self.processed_paths[0])

    def __repr__(self) -> str:
        arg_repr = str(len(self)) if len(self) > 1 else ''
        return f'{self.__class__.__name__}({arg_repr}) {self.name}'
