""" Create a dataset with HeteroData format.
"""
import json
import os
import os.path as osp
from glob import glob
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData, InMemoryDataset

from py_script.utils import create_dir, read_mpc


class GMD(InMemoryDataset):
    """ GMD dataset.

    Args:
        root (Optional[str], optional): The root folder for data to be stored. Defaults to './'.
        names (Optional[str], optional): A list of names of grids. Defaults to "b4gic".
        problem (Optional[str], optional): Specify the problem to solve. Defaults to 'clf'.
        setting (Optional[str], optional): Specify the setting to solve in. Defaults to 'gic'.
        force_reprocess (Optional[bool], optional): Force to reprocess data if `True`. Defaults to False.
        transform (Optional[Callable], optional): Transfom modules. Defaults to None.
        pre_transform (Optional[Callable], optional): Pre_transform modules. Defaults to None.
        pre_filter (Optional[Callable], optional): Pre_filter modules. Defaults to None.
    """

    def __init__(self, root: Optional[str] = "./",
                 names: Optional[list] = ["epri21"],
                 setting: Optional[str] = 'gic',
                 problem: Optional[str] = 'reg',
                 force_reprocess: Optional[bool] = False,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):

        self.root = root
        self.name = names[0]    # self.name seems to be needed for processed paths.
                                # I'm not sure how to deal with this, so I'll use
                                # --force for now.
        self.test_grids = names
        self.setting = setting
        self.transform = transform
        self.force_reprocess = force_reprocess
        self.problem = problem
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        for grid in self.test_grids:
            self.mpc_file = osp.join(osp.dirname(osp.realpath(__file__)), "..", "test", "data", f"{grid}.m")

            if self.force_reprocess:
                SAVED_PATH = osp.join(osp.abspath(self.root), "processed", grid)
                SAVED_FILE = f"{SAVED_PATH}/processed.pt"
                if osp.exists(SAVED_FILE):
                    os.remove(SAVED_FILE)

        # self.mpc_file = osp.join(osp.dirname(osp.realpath(__file__)), "..", "test", "data", f"{self.name}.m")

        # if self.force_reprocess:
        #     SAVED_PATH = osp.join(osp.abspath(self.root), "processed", self.name)
        #     SAVED_FILE = f"{SAVED_PATH}/processed.pt"
        #     if osp.exists(SAVED_FILE):
        #         os.remove(SAVED_FILE)

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        SAVED_PATH = osp.join(osp.abspath(self.root), "processed", self.name)
        create_dir(SAVED_PATH)
        return [f'{SAVED_PATH}/processed.pt']

    def process(self):
        """ Process the raw file, and save to processed files. """
        data_list = []

        for grid in self.test_grids:
            # enumerate the optimized file, send into a list
            if self.setting == "mld":
                res_files = glob(f"../gic-blockers/results/{grid}_*.json")
            else:
                res_files = glob("./gic_blocker_results/*.json")

            for res_f in res_files:
                # For mld only, read in the mods files
                if self.setting == "mld":
                    id = res_f[-9:-5]
                    mods_file = f"../gic-blockers/mods/{grid}_{id}.json"
                    mods_load = json.load(open(mods_file))

                # Modded version
                res_data = json.load(open(res_f))
                # read the matpower file
                # TODO: make the data processing more efficiently, no need to read the MPC in every loop
                mpc = read_mpc(self.mpc_file)

                # For mld only, store the number of busses in the power grid
                if self.setting == "mld":
                    n_bus = mpc['bus'].shape[0]
                mapping = dict((v, k) for k, v in mpc['bus'].bus_i.items())

                # For mld only, check to see if the power grid results are infeasible
                if (self.setting == "mld") and ("INFEASIBLE" in res_data['termination_status']):
                    pass

                h_data = HeteroData()

                # Different code for handling nodes in the heterogeneous graph neural network
                if self.setting == "mld":
                    case_load = mods_load['load']
                    res_load = res_data['solution']['load']

                    load_bus_idx = [case_load[load_idx]['source_id'][1] for load_idx in case_load]
                    h_data.load_bus_mask = torch.zeros(n_bus).bool()
                    h_data.load_bus_mask[load_bus_idx] = True

                    # a dict from bus_i to load_idx
                    map_bus_to_load = {case_load[load_idx]['source_id'][1]: load_idx for load_idx in case_load}

                    # Stores all the bus_i indices from the "load_bus" variable (basically the
                    # aligned keys). Used for extracting the results.
                    for k in map_bus_to_load:
                        # update pd/qd with bus_i
                        mpc['bus'].loc[mpc['bus']['bus_i'] == int(k),
                                    "Pd"] = case_load[map_bus_to_load[k]]['pd'] * 100
                        mpc['bus'].loc[mpc['bus']['bus_i'] == int(k),
                                    "Qd"] = case_load[map_bus_to_load[k]]['qd'] * 100

                    if self.problem == "clf":
                        y = [res_load[map_bus_to_load[k]]['status']
                            for k in sorted(list(map_bus_to_load.keys()))]
                        h_data['y'] = torch.tensor(np.array(y).round(), dtype=torch.long)
                    else:
                        y = [res_load[map_bus_to_load[k]]['qd']
                            for k in sorted(list(map_bus_to_load.keys()))]
                        # y = [res_load[map_bus_to_load[k]]['status']
                        #      for k in sorted(list(map_bus_to_load.keys()))]
                        h_data['y'] = torch.tensor(np.array(y).reshape(-1, 1), dtype=torch.float32)
                else:
                    # Stores all the bus_i indices from the "load_bus" variable (basically the
                    # aligned keys). Used for extracting the results.
                    for k in res_data['bus'].keys():
                        # update pd/qd with bus_i
                        mpc['bus'].loc[mpc['bus']['bus_i'] == int(k), "Pd"] = res_data['bus'][k]['pd']
                        mpc['bus'].loc[mpc['bus']['bus_i'] == int(k), "Qd"] = res_data['bus'][k]['qd']

                    if self.problem == "clf":
                        y = [int(res_data['gmd_bus'][k]['gic_blocker']) for k in sorted(list(res_data['gmd_bus'].keys()))]
                        h_data['y'] = torch.tensor(np.array(y).round(), dtype=torch.long)
                    else:
                        y = [int(res_data['gmd_bus'][k]['volt_mag']) for k in sorted(list(res_data['gmd_bus'].keys()))]
                        h_data['y'] = torch.tensor(np.array(y).reshape(-1, 1), dtype=torch.float32)

                ''' node_type: bus '''
                # convert the `type` to one-hot encoder
                mpc['bus'] = pd.concat([mpc['bus'], pd.DataFrame(
                    np.eye(4)[mpc['bus'].type.to_numpy(dtype=int)]).add_prefix("t")], axis=1)
                mpc['bus'] = mpc['bus'].drop(['type'], axis=1)
                h_data['bus'].x = torch.tensor(mpc['bus'].iloc[:, 1:].to_numpy(), dtype=torch.float32)

                # Store the number of nodes to use as the input into the forward function. Don't use num_nodes because
                # it's a PyTorch variable that stores the total number of nodes, regardless of the type.
                h_data.num_network_nodes = mpc['bus'].shape[0]

                # extract the node_i from bus_i for perturbed load
                # node_idx_y = [mapping[k] for k in map_bus_to_load]

                ''' node_type: gen '''
                # creating new virtual link between bus and gen to handle multiple generators
                h_data['gen'].x = torch.tensor(mpc['gen'].iloc[:, 1:].to_numpy(), dtype=torch.float32)

                ''' edge_type (virtual): gen--conn--bus '''
                n_gen = mpc['gen'].shape[0]

                gen_bus_edges = np.zeros((n_gen, 2))
                for i in range(n_gen):
                    gen_bus_edges[i] = [mapping[mpc['gen'].bus[i]], i]
                # edge feature
                h_data['bus', 'conn', "gen"].edge_index = torch.tensor(gen_bus_edges.T, dtype=torch.long)

                gen_bus_edges = np.zeros((n_gen, 2))
                for i in range(n_gen):
                    gen_bus_edges[i] = [i, mapping[mpc['gen'].bus[i]]]
                # edge feature
                h_data['gen', 'conn', "bus"].edge_index = torch.tensor(gen_bus_edges.T, dtype=torch.long)

                # convert the tuples with mapping
                ''' edge_type: bus--branch--bus '''
                n_branch = mpc['branch'].shape[0]
                edges = np.zeros((n_branch, 2))
                for i in range(n_branch):
                    edges[i] = [mapping[mpc['branch'].fbus[i]], mapping[mpc['branch'].tbus[i]]]
                h_data['bus', 'branch', 'bus'].edge_index = torch.tensor(edges.T, dtype=torch.long)
                h_data['bus', 'branch', 'bus'].edge_attr = torch.tensor(
                    mpc['branch'].iloc[:, 2:].to_numpy(), dtype=torch.float32)

                ''' edge_type: bus--branch_gmd--bus '''
                # convert type and config to one-hot encoder
                bg_type = {"'xfmr'": 0, "'line'": 1, "'series_cap'": 2}
                bg_config = {"'none'": 0, "'delta-delta'": 1, "'delta-wye'": 2,
                            "'wye-delta'": 3, "'wye-wye'": 4, "'delta-gwye'": 5,
                            "'gwye-delta'": 6, "'gwye-gwye'": 7, "'gwye-gwye-auto'": 8}
                mpc['branch_gmd']['type'] = mpc['branch_gmd']['type'].map(lambda x: bg_type[x])
                mpc['branch_gmd'] = pd.concat([mpc['branch_gmd'], pd.DataFrame(
                    np.eye(3)[mpc['branch_gmd']['type'].to_numpy(dtype=int)]).add_prefix("t")], axis=1)

                mpc['branch_gmd']['config'] = mpc['branch_gmd']['config'].map(lambda x: bg_config[x])
                mpc['branch_gmd'] = pd.concat([mpc['branch_gmd'], pd.DataFrame(
                    np.eye(9)[mpc['branch_gmd']['config'].to_numpy(dtype=int)]).add_prefix("c")], axis=1)
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
                ''' node_type: gmd_bus '''
                # NOTE: only read GMD from conf file
                h_data['gmd_bus'].x = torch.tensor(mpc['gmd_bus'].iloc[:, 1:3].to_numpy(), dtype=torch.float32)

                ''' edge_type: gmd_bus--gmd_branch--gmd_bus '''
                gmd_edges = mpc['gmd_branch'].iloc[:, :2].to_numpy()
                # gmd edge index
                h_data['gmd_bus', 'gmd_branch', 'gmd_bus'].edge_index = torch.tensor(gmd_edges.T - 1, dtype=torch.long)
                h_data['gmd_bus', 'gmd_branch', 'gmd_bus'].edge_attr = torch.tensor(
                    mpc['gmd_branch'].iloc[:, 3:-1].to_numpy(), dtype=torch.float32)

                ''' edge_type (virtual): gmd_bus--attach--bus '''
                n_gmd_bus = mpc['gmd_bus'].shape[0]
                gmd_bus_bus_edges = np.zeros((n_gmd_bus, 2))
                for i in range(n_gmd_bus):
                    gmd_bus_bus_edges[i] = [i, mapping[mpc['gmd_bus'].parent_index[i]]]

                h_data['gmd_bus', 'attach', "bus"].edge_index = torch.tensor(gmd_bus_bus_edges.T, dtype=torch.long)

                h_data = h_data if self.pre_transform is None else self.pre_transform(h_data)
                data_list.append(h_data)

        # save to the processed path
        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self) -> str:
        arg_repr = str(len(self)) if len(self) > 1 else ''
        return f'{self.__class__.__name__}({arg_repr}) {self.name}'