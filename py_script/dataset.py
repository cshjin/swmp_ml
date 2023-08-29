""" Create a dataset with HeteroData format. """
import json
import os
import os.path as osp
import pickle
from copy import deepcopy
from glob import glob
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch, HeteroData, InMemoryDataset

from py_script.utils import create_dir, preprocess_mpc, read_mpc


class GMD(InMemoryDataset):
    """ GMD dataset.

    Args:
        root (Optional[str], optional):                 The root folder for data to be stored. Defaults to './'.
        name (Optional[str], optional):                 Name of grid. Defaults to "b4gic".
        setting (Optional[str], optional):              Specify the setting to solve. Defaults to 'gic'.
        force_reprocess (Optional[bool], optional):     Force to reprocess data if `True`. Defaults to False.
        transform (Optional[Callable], optional):       Transfom modules. Defaults to None.
        pre_transform (Optional[Callable], optional):   Pre_transform modules. Defaults to None.
        pre_filter (Optional[Callable], optional):      Pre_filter modules. Defaults to None.
    """

    def __init__(self, root: Optional[str] = "./",
                 name: Optional[str] = "epri21",
                 setting: Optional[str] = 'gic',
                 force_reprocess: Optional[bool] = False,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 log: bool = True):

        self.root = root
        self.name = name
        self.setting = setting
        self.force_reprocess = force_reprocess
        self.transform = transform
        self.problem = "clf" if self.setting == "gic" else "reg"
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        # input case file
        self._INPUT_FILE = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "matpower", f"{self.name}.m")
        # dir of processed data
        self._SAVED_PATH = osp.join(osp.abspath(self.root), "processed", self.name)
        # create_dir if not exist
        create_dir(self._SAVED_PATH)
        # path of processed data
        self._SAVED_FILE = f"{self._SAVED_PATH}/processed.pt"

        if self.force_reprocess:
            # remove old procesed files
            if osp.exists(self._SAVED_FILE):
                os.remove(self._SAVED_FILE)

        super().__init__(root, transform, pre_transform, pre_filter, log)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> List[str]:
        r""" Names of preprocessed files.

        Returns:
            List[str]: List of processed files.
                       First is the one of pt format processed for PyTorch.
        """
        return [f"{self._SAVED_PATH}/processed.pt",
                f"{self._SAVED_PATH}/data_list.pkl"]

    @property
    def raw_file_names(self) -> List[str]:
        r""" Names of raw files.

        Returns:
            List[str]: List of raw files.
        """
        # TODO: move the data into `data` folder
        if self.setting == "mld":
            # regression problem
            raw_files = glob(f"../gic-blockers/results/{self.name}_*.json")
        else:
            # classification problem
            curdir = osp.dirname(osp.realpath(__file__))
            raw_path = osp.join(curdir, "..", "data", "gic", "results", self.name)
            raw_files = glob(f"{raw_path}/*_{self.name}_*.json")
        return raw_files

    def process(self):
        r"""Process the raw data to the processed data in pytorch geometric. """

        # read the MATPOWER case file
        mpc = read_mpc(self._INPUT_FILE)
        mpc = preprocess_mpc(mpc)
        # h_data = convert_hdata(mpc)
        if self.setting == "mld":
            data_list = process_mld_files(files=self.raw_file_names,
                                          name=self.name,
                                          ori_mpc=mpc,
                                          pre_transform=self.pre_transform)
        elif self.setting == "gic":
            data_list = process_gic_files(files=self.raw_file_names,
                                          mpc=mpc,
                                          #  data=h_data,
                                          pre_transform=self.pre_transform)
        else:
            raise ValueError(f"Unknown setting: {self.setting}")

        # pickle the data_list
        pickle.dump(data_list, open(self.processed_paths[1], 'wb'))
        # save to the processed path
        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self) -> str:
        r""" Customize the print of the dataset.

        Returns:
            str: the print of the dataset
        """
        arg_repr = str(len(self)) if len(self) > 1 else ''
        return f'{self.__class__.__name__}({arg_repr}) {self.name}'


class MultiGMD(InMemoryDataset):

    def __init__(self, root: Optional[str] = None,
                 names: Optional[list] = ['epri21'],
                 setting: Optional[str] = 'gic',
                 transform: Optional[Callable] = None,
                 force_reprocess: Optional[bool] = False,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):

        self.root = root
        self.names = names
        self.setting = setting
        self.transform = transform
        self.force_reprocess = force_reprocess
        self.problem = "clf" if self.setting == "gic" else "reg"
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        for name in self.names:
            dataset = GMD(root=self.root,
                          name=name,
                          setting=self.setting,
                          transform=self.transform,
                          force_reprocess=self.force_reprocess,
                          pre_transform=self.pre_transform,
                          pre_filter=self.pre_filter)
        if self.force_reprocess:
            SAVED_PATH = osp.join(osp.abspath(self.root), "processed", "multi_gmd")
            SAVED_FILE = f"{SAVED_PATH}/processed.pt"
            if osp.exists(SAVED_FILE):
                os.remove(SAVED_FILE)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        dir_path = osp.abspath(self.root)
        SAVED_PATH = osp.join(dir_path, "processed", "multi_gmd")
        create_dir(SAVED_PATH)
        return [f'{SAVED_PATH}/processed.pt']

    def process(self):
        """ Process multiple grids into single dataset in PyG
        """
        # DEBUG: build multi_gmd data_list with diff samples from diff grids
        data_list = []
        dir_path = osp.abspath(self.root)
        # for name in self.names:
        #     data_path = osp.join(dir_path, "processed", name)
        #     data = torch.load(f"{data_path}/processed.pt")[0]
        #     data_list.append(data)

        for name in self.names:
            data_path = osp.join(dir_path, "processed", name)
            _data = pickle.load(open(f"{data_path}/data_list.pkl", 'rb'))
            data_list += _data

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def process_mld_files(files, name, ori_mpc, problem="reg", pre_transform=None, pre_filter=None, **kwargs):
    # TODO: need to be updated
    r""" Process raw files into PyG data format for MLD problem.

    Args:
        files (List[str]): List of raw files.
        name (str): Name of the dataset.
        ori_mpc (pd.DataFrame): Pandas dataframe of the grid.
        problem (str, optional): Problem type. Defaults to "reg".
        pre_transform (callable, optional): Pre_transform function. Defaults to None.
        pre_filter (callable, optional): Pre_filter function. Defaults to None.

    Returns:
        List[Data]: List of PyG data.
    """
    # string mapping of `branch_gmd` `type` and `config`
    bg_type = {"'xfmr'": 0, "'line'": 1, "'series_cap'": 2}
    bg_config = {"'none'": 0, "'delta-delta'": 1, "'delta-wye'": 2,
                 "'wye-delta'": 3, "'wye-wye'": 4, "'delta-gwye'": 5,
                 "'gwye-delta'": 6, "'gwye-gwye'": 7, "'gwye-gwye-auto'": 8}
    data_list = []
    # process mpc dataframe
    n_bus = mpc["bus"].shape[0]
    bus_id_idx = {(v, k) for k, v in mpc["bus"].bus_i.items()}

    # process raw files one by one
    for f in files:
        mpc = deepcopy(ori_mpc)

        id = f[-9:-5]
        mods_file = f"../gic-blockers/mods/{name}_{id}.json"
        mods_load = json.load(open(mods_file))

        raw_data = json.load(open(f))

        # bypass the infeasible cases
        if "INFEASIBLE" in raw_data["termination_status"]:
            pass

        # create the heterogeneous data
        h_data = HeteroData()

        case_load = mods_load["load"]
        res_load = raw_data["solution"]["load"]

        # load bus mask
        load_bus_idx = [case_load[load_idx]['source_id'][1] for load_idx in case_load]
        h_data.load_bus_mask = torch.zeros(n_bus).bool()
        h_data.load_bus_mask[load_bus_idx] = True

        map_bus_to_load = {case_load[load_idx]['source_id'][1]: load_idx for load_idx in case_load}

        for k in map_bus_to_load:
            # update pd/qd with bus_i
            mpc["bus"].loc[mpc["bus"]["bus_i"] == int(k),
                           "Pd"] = case_load[map_bus_to_load[k]]["pd"] * float(mpc["baseMVA"])
            mpc["bus"].loc[mpc["bus"]["bus_i"] == int(k),
                           "Qd"] = case_load[map_bus_to_load[k]]["qd"] * float(mpc["baseMVA"])

        if problem == "clf":
            try:
                y = [int(raw_data['gmd_bus'][k]['gic_blocker'])
                     for k in sorted(list(raw_data['gmd_bus'].keys()))]
                h_data['y'] = torch.tensor(np.array(y).round(), dtype=torch.long)
            except Exception:
                print("Null in result file")
                continue
        else:
            y = [int(raw_data['gmd_bus'][k]['volt_mag']) for k in sorted(list(raw_data['gmd_bus'].keys()))]
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
            gen_bus_edges[i] = [bus_id_idx[mpc['gen'].bus[i]], i]
        # edge feature
        h_data['bus', 'conn', "gen"].edge_index = torch.tensor(gen_bus_edges.T, dtype=torch.long)

        gen_bus_edges = np.zeros((n_gen, 2))
        for i in range(n_gen):
            gen_bus_edges[i] = [i, bus_id_idx[mpc['gen'].bus[i]]]
        # edge feature
        h_data['gen', 'conn', "bus"].edge_index = torch.tensor(gen_bus_edges.T, dtype=torch.long)

        # convert the tuples with mapping
        ''' edge_type: bus--branch--bus '''
        n_branch = mpc['branch'].shape[0]
        edges = np.zeros((n_branch, 2))
        for i in range(n_branch):
            edges[i] = [bus_id_idx[mpc['branch'].fbus[i]], bus_id_idx[mpc['branch'].tbus[i]]]
        h_data['bus', 'branch', 'bus'].edge_index = torch.tensor(edges.T, dtype=torch.long)
        h_data['bus', 'branch', 'bus'].edge_attr = torch.tensor(
            mpc['branch'].iloc[:, 2:].to_numpy(), dtype=torch.float32)

        ''' edge_type: bus--branch_gmd--bus '''

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
            edges[i] = [bus_id_idx[mpc['branch_gmd'].hi_bus[i]], bus_id_idx[mpc['branch_gmd'].lo_bus[i]]]
        h_data['bus', 'branch_gmd', 'bus'].edge_index = torch.tensor(edges.T, dtype=torch.long)
        h_data['bus', 'branch_gmd', 'bus'].edge_attr = torch.tensor(
            mpc['branch_gmd'].iloc[:, 2:].to_numpy(), dtype=torch.float32)

        pos = mpc['bus_gmd'].to_numpy()
        pos = torch.tensor(pos, dtype=torch.float32)

        ''' DC network with GMD data '''
        ''' node_type: gmd_bus '''
        # NOTE: only read GMD from conf file
        h_data['gmd_bus'].x = torch.tensor(mpc['gmd_bus'].iloc[:, 1:3].to_numpy(), dtype=torch.float32)

        # NOTE: gic blocker bus mask, those are candidate gmd buses for gic blockers
        n_gmd_bus = mpc['gmd_bus'].shape[0]
        h_data.gic_blocker_bus_mask = torch.zeros(n_gmd_bus).bool()
        gic_blocker_bus_idx = np.array(mpc['gmd_bus'][mpc['gmd_bus']['g_gnd'] > 0].index.tolist())
        h_data.gic_blocker_bus_mask[gic_blocker_bus_idx] = True

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
            gmd_bus_bus_edges[i] = [i, bus_id_idx[mpc['gmd_bus'].parent_index[i]]]

        h_data['gmd_bus', 'attach', "bus"].edge_index = torch.tensor(gmd_bus_bus_edges.T, dtype=torch.long)

        h_data = h_data if pre_transform is None else pre_transform(h_data)
        data_list.append(h_data)

    return data_list


def process_gic_files(files, mpc, pre_transform=None, pre_filter=None, **kwargs):
    data_list = []

    h_data = HeteroData()

    """ AC Network """
    ''' node_type: bus '''
    h_data['bus'].x = torch.tensor(mpc['bus'].iloc[:, 1:].to_numpy(), dtype=torch.float32)
    # store the column names
    h_data['bus'].x_names = mpc['bus'].columns[1:].tolist()

    ''' node_type: gen '''
    h_data['gen'].x = torch.tensor(mpc['gen'].iloc[:, 1:].to_numpy(), dtype=torch.float32)

    ''' edge_type (virtual): gen--conn--bus '''
    n_gen = mpc['gen'].shape[0]
    # from gen to bus
    gen_bus_edges = np.zeros((n_gen, 2))
    for i, row in mpc['gen'].iterrows():
        gen_bus_edges[i] = [i, row['bus']]
    h_data['gen', 'conn', 'bus'].edge_index = torch.tensor(gen_bus_edges.T, dtype=torch.long)
    # NOTE: from bus to gen
    bus_gen_edges = np.zeros((n_gen, 2))
    for i, row in mpc['gen'].iterrows():
        bus_gen_edges[i] = [row['bus'], i]
    h_data['bus', 'conn', 'gen'].edge_index = torch.tensor(bus_gen_edges.T, dtype=torch.long)

    ''' edge_type: bus--branch--bus '''
    n_branch = mpc['branch'].shape[0]
    branch_edges = np.zeros((n_branch, 2))
    for i, row in mpc['branch'].iterrows():
        branch_edges[i] = [row['fbus'], row['tbus']]
    h_data['bus', 'branch', 'bus'].edge_index = torch.tensor(branch_edges.T, dtype=torch.long)
    h_data['bus', 'branch', 'bus'].edge_attr = torch.tensor(mpc['branch'].iloc[:, 2:].to_numpy(), dtype=torch.float32)
    h_data['bus', 'branch', 'bus'].edge_attr_names = mpc['branch'].columns[2:].tolist()

    ''' edge_type: bus--branch_gmd--bus '''
    n_branch_gmd = mpc['branch_gmd'].shape[0]
    branch_gmd_edges = np.zeros((n_branch_gmd, 2))
    for i, row in mpc['branch_gmd'].iterrows():
        branch_gmd_edges[i] = [row['hi_bus'], row['lo_bus']]
    h_data['bus', 'branch_gmd', 'bus'].edge_index = torch.tensor(branch_gmd_edges.T, dtype=torch.long)
    h_data['bus', 'branch_gmd', 'bus'].edge_attr = torch.tensor(
        mpc['branch_gmd'].iloc[:, 4:].to_numpy(), dtype=torch.float32)
    h_data['bus', 'branch_gmd', 'bus'].edge_attr_names = mpc['branch_gmd'].columns[4:].tolist()

    # bus geo data
    # NOTE: remove the type bus_gmd
    # h_data['bus_gmd'].x = torch.tensor(mpc['bus_gmd'].to_numpy(), dtype=torch.float32)

    """ DC Network """
    ''' node_type: gmd_bus '''
    h_data['gmd_bus'].x = torch.tensor(mpc['gmd_bus'].iloc[:, 1:3].to_numpy(), dtype=torch.float32)

    n_gmd_bus = mpc['gmd_bus'].shape[0]
    h_data['substation_mask'] = torch.zeros(n_gmd_bus).bool()
    substation_index = np.array(mpc['gmd_bus'][mpc['gmd_bus']['g_gnd'] > 0].index.tolist())
    h_data['substation_mask'][substation_index] = True

    ''' edge_type: gmd_bus--attach--bus '''
    n_gmd_bus = mpc['gmd_bus'].shape[0]
    gmd_bus_bus_edges = np.zeros((n_gmd_bus, 2))
    for i, row in mpc['gmd_bus'].iterrows():
        gmd_bus_bus_edges[i] = [i, row['parent_index']]
    h_data['gmd_bus', 'attach', "bus"].edge_index = torch.tensor(gmd_bus_bus_edges.T, dtype=torch.long)

    ''' edge_type: gmd_bus--gmd_branch--gmd_bus'''
    n_gmd_branch = mpc['gmd_branch'].shape[0]
    gmd_branch_edges = np.zeros((n_gmd_branch, 2))
    for i, row in mpc['gmd_branch'].iterrows():
        gmd_branch_edges[i] = [row['f_bus'], row['t_bus']]
    h_data['gmd_bus', 'gmd_branch', 'gmd_bus'].edge_index = torch.tensor(gmd_branch_edges.T, dtype=torch.long)
    h_data['gmd_bus', 'gmd_branch', 'gmd_bus'].edge_attr = torch.tensor(
        mpc['gmd_branch'].iloc[:, 3:-1].to_numpy(), dtype=torch.float32)
    h_data['gmd_bus', 'gmd_branch', 'gmd_bus'].edge_attr_names = mpc['gmd_branch'].columns[3:-1].tolist()

    # Modify the data based on simulations (x, y)
    for f in files:
        _h_data = deepcopy(h_data)
        res_data = json.load(open(f))

        for k in res_data['bus'].keys():
            idx = mpc['bus'].loc[mpc['bus']['bus_i'] == int(k)].index[0]
            _h_data['bus'].x[idx, 0] = res_data['bus'][k]['pd'] * mpc['baseMVA']
            _h_data['bus'].x[idx, 1] = res_data['bus'][k]['qd'] * mpc['baseMVA']

        try:
            y = [int(res_data['gmd_bus'][k]['gic_blocker']) for k in sorted(list(res_data['gmd_bus'].keys()))]
            _h_data['y'] = torch.tensor(np.array(y).round(), dtype=torch.long)
        except Exception:
            print("Null in result file. infeasible solution.")
            continue

        data_list.append(_h_data)

    return data_list


def convert_hdata(mpc):
    """ Convert the MPC dict to heterogeneous data format.

    Args:
        mpc (dict): Dictionary of processed MatPower data.

    Returns:
        pyg.Data.HeteroData: Heterogeneous data format.
    """

    mpc = deepcopy(mpc)
    h_data = HeteroData()

    """ AC Network """
    ''' node_type: bus '''
    h_data['bus'].x = torch.tensor(mpc['bus'].iloc[:, 1:].to_numpy(), dtype=torch.float32)
    # store the column names
    h_data['bus'].x_names = mpc['bus'].columns[1:].tolist()

    ''' node_type: gen '''
    h_data['gen'].x = torch.tensor(mpc['gen'].iloc[:, 1:].to_numpy(), dtype=torch.float32)

    ''' edge_type (virtual): gen--conn--bus '''
    n_gen = mpc['gen'].shape[0]
    # from gen to bus
    gen_bus_edges = np.zeros((n_gen, 2))
    for i, row in mpc['gen'].iterrows():
        gen_bus_edges[i] = [row['bus'], i]
    h_data['gen', 'conn', 'bus'] = torch.tensor(gen_bus_edges.T, dtype=torch.long)
    # NOTE: from bus to gen
    # bus_gen_edges = np.zeros((n_gen, 2))
    # for i, row in mpc['gen'].iterrows():
    #     bus_gen_edges[i] = [i, row['bus']]
    # h_data['bus', 'conn', 'gen'] = torch.tensor(bus_gen_edges.T, dtype=torch.long)

    ''' edge_type: bus--branch--bus '''
    n_branch = mpc['branch'].shape[0]
    branch_edges = np.zeros((n_branch, 2))
    for i, row in mpc['branch'].iterrows():
        branch_edges[i] = [row['fbus'], row['tbus']]
    h_data['bus', 'branch', 'bus'].edge_index = torch.tensor(branch_edges.T, dtype=torch.long)
    h_data['bus', 'branch', 'bus'].edge_attr = torch.tensor(mpc['branch'].iloc[:, 2:].to_numpy(), dtype=torch.float32)
    h_data['bus', 'branch', 'bus'].edge_attr_names = mpc['branch'].columns[2:].tolist()

    ''' edge_type: bus--branch_gmd--bus '''
    n_branch_gmd = mpc['branch_gmd'].shape[0]
    branch_gmd_edges = np.zeros((n_branch_gmd, 2))
    for i, row in mpc['branch_gmd'].iterrows():
        branch_gmd_edges[i] = [row['hi_bus'], row['lo_bus']]
    h_data['bus', 'branch_gmd', 'bus'].edge_index = torch.tensor(branch_gmd_edges.T, dtype=torch.long)
    h_data['bus', 'branch_gmd', 'bus'].edge_attr = torch.tensor(
        mpc['branch_gmd'].iloc[:, 4:].to_numpy(), dtype=torch.float32)
    h_data['bus', 'branch_gmd', 'bus'].edge_attr_names = mpc['branch_gmd'].columns[4:].tolist()

    # bus geo data
    h_data['bus_gmd'].x = torch.tensor(mpc['bus_gmd'].to_numpy(), dtype=torch.float32)

    """ DC Network """
    ''' node_type: gmd_bus '''
    h_data['gmd_bus'].x = torch.tensor(mpc['gmd_bus'].iloc[:, 1:3].to_numpy(), dtype=torch.float32)

    n_gmd_bus = mpc['gmd_bus'].shape[0]
    h_data['substation_mask'] = torch.zeros(n_gmd_bus).bool()
    substation_index = np.array(mpc['gmd_bus'][mpc['gmd_bus']['g_gnd'] > 0].index.tolist())
    h_data['substation_mask'][substation_index] = True

    ''' edge_type: gmd_bus--attach--bus '''
    n_gmd_bus = mpc['gmd_bus'].shape[0]
    gmd_bus_bus_edges = np.zeros((n_gmd_bus, 2))
    for i, row in mpc['gmd_bus'].iterrows():
        gmd_bus_bus_edges[i] = [i, row['parent_index']]
    h_data['gmd_bus', 'attach', "bus"].edge_index = torch.tensor(gmd_bus_bus_edges.T, dtype=torch.long)

    ''' edge_type: gmd_bus--gmd_branch--gmd_bus'''
    n_gmd_branch = mpc['gmd_branch'].shape[0]
    gmd_branch_edges = np.zeros((n_gmd_branch, 2))
    for i, row in mpc['gmd_branch'].iterrows():
        gmd_branch_edges[i] = [row['f_bus'], row['t_bus']]
    h_data['gmd_bus', 'gmd_branch', 'gmd_bus'].edge_index = torch.tensor(gmd_branch_edges.T, dtype=torch.long)
    h_data['gmd_bus', 'gmd_branch', 'gmd_bus'].edge_attr = torch.tensor(
        mpc['gmd_branch'].iloc[:, 3:-1].to_numpy(), dtype=torch.float32)
    h_data['gmd_bus', 'gmd_branch', 'gmd_bus'].edge_attr_names = mpc['gmd_branch'].columns[3:-1].tolist()

    return h_data
