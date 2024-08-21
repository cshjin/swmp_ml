""" Utility functions for data processing and model training.

Reference:
* [Matpower Manual: v7.1]: http://matpower.org/docs/MATPOWER-manual.pdf

Author: SWMP-Team
License: MIT
"""
import os
import re
from io import StringIO

import pandas as pd


HEADERS = {
    "bus": ['bus_i', 'type', 'Pd', 'Qd', 'Gs', 'Bs', 'area',
            'Vm', 'Va', 'baseKV', 'zone', 'Vmax', 'Vmin'],
    "gen": ['bus', 'Pg', 'Qg', 'Qmax', 'Qmin',
            'Vg', 'mBase', 'status', 'Pmax', 'Pmin',
            'Pc1', 'Pc2', 'Qc1min', 'Qc1max',
            'Qc2min', 'Qc2max', 'ramp_agc', 'ramp_10', 'ramp_30', 'ramp_q', 'apf'],
    "gencost": ['model', 'startup', 'shutdown', 'ncost', 'c0', 'c1', 'c2'],
    "branch": ['fbus', 'tbus', 'r', 'x', 'b',
               'rateA', 'rateB', 'rateC', 'ratio',
               'angle', 'status', 'angmin', 'angmax',
               'Pf', 'Qf', 'Pt', 'Qt'],
    "gmd_bus": ['parent_index', 'status', 'g_gnd', 'sub', 'name'],
    "gmd_branch": ['f_bus', 't_bus', 'parent_type', 'parent_index', 'br_status',
                   'br_r', 'br_v', 'len_km', 'name'],
    "branch_gmd": ['hi_bus', 'lo_bus', 'gmd_br_hi', 'gmd_br_lo',
                   'gmd_k', 'gmd_br_series', 'gmd_br_common', 'baseMVA', 'type', 'config'],
    "branch_thermal": ['xfmr', 'temperature_ambient', 'hotspot_instant_limit',
                       'hotspot_avg_limit', 'hotspot_rated', 'topoil_time_const', 'topoil_rated',
                       'topoil_init', 'topoil_initialized', 'hotspot_coeff'],
    "bus_gmd": ['lat', 'lon'],
}


def clean_file(input_file):
    r""" Clean the input matpower file, and save as {input_file}_cleaned.m in the same folder.

    - replace one or more tabs in the file with a single space
    - remove leading or padding tabs and spaces of each line
    - some blocks of the file are using `{`, replace them with `[`, similarly, replace `}` with `]`
    - file ends with a newline character

    Args:
        input_file (str): The input file to clean.

    Returns:
        str: The output file name.
    """
    # Read the original file
    try:
        with open(input_file, 'r') as f:
            content = f.readlines()
    except Exception as e:
        print("Error: {}".format(e))
        exit(-1)

    # Replace one or more tabs with a single space, remove leading and trailing spaces or tabs from each line,
    # and replace {} with []
    cleaned_content = [re.sub('\\s+', ' ', line).strip().replace('{', '[').replace('}', ']') for line in content]

    # Ensure file ends with a newline character
    cleaned_content.append('')

    # Join the cleaned lines back into a single string
    cleaned_content = '\n'.join(cleaned_content)

    # Create the new filename
    base, ext = os.path.splitext(input_file)
    output_file = f"{base}_cleaned{ext}"

    # Write the cleaned content to the new file
    try:
        with open(output_file, 'w') as f:
            f.write(cleaned_content)
    except Exception as e:
        print("Error: {}".format(e))
        exit(-1)

    return output_file


def read_mpc(fn):
    r""" Read the MPC MATPOWER file ".m" into dictionary.
    Args:
        fn (str): Filename.

    Returns:
        dict: A dictionary with keys of attributes.

    References:
        [1]: MATPOWER manual (https://matpower.org/docs/MATPOWER-manual.pdf), Appendix B.
    """
    fn = clean_file(fn)
    mpc = {}
    with open(fn, "r") as f:
        string = f.read()

    lines = []
    with open(fn, "r") as f:
        for line in f:
            lines.append(line)

    matches = re.findall(r"mpc\.\w+", string)
    for attr in matches:
        key = attr.split(".")[1]

        # line_number = string.count('\n', 0, string.index(attr))
        # if line_number > 0:
        #     header_line = lines[line_number - 1]
        #     print(header_line)
        # process with different patterns
        if key in ['version', 'baseMVA', 'time_elapsed']:
            # the key with only one value
            pattern = rf'mpc\.{key}\s*=\s*(?P<data>.*?);'
            match = re.search(pattern, string, re.DOTALL)
            value = match.groupdict()['data'].strip("'").strip('"')
            # if key == 'baseMVA' or key == 'time_elapsed':
            #     value = float(value)
            mpc[key] = value

        elif key in ['gen', 'bus', 'branch', 'branch_gmd', 'bus_gmd']:
            # DEPRECATED: ignore the header line above the data
            # line_number = string.count('\n', 0, string.index(attr))
            # if line_number > 0:
            #     header_line = lines[line_number - 1]
            #     header = header_line.strip().split("%")[-1].strip().split()

            # the keys with standard MATPOWER data
            pattern = rf'mpc\.{key}\s*=\s*\[[\n]?(?P<data>.*?)[\n]?\];'
            match = re.search(pattern, string, re.DOTALL)
            # convert to string for pandas dataframe
            data = StringIO(match.groupdict()['data'])
            # df = pd.read_csv(data, sep="\t", header=None)
            # after processing file spaces only separators
            df = pd.read_csv(data, sep="\\s+", header=None, comment="%", engine="python")

            if df.shape[1] == len(HEADERS[key]):
                df.columns = HEADERS[key]
            elif df.shape[1] < len(HEADERS[key]):
                # fill the missing columns with 0
                df.columns = HEADERS[key][:df.shape[1]]
                for col in HEADERS[key][df.shape[1]:]:
                    df[col] = 0

            # drop the first column
            # if key == "bus":
            #     df = df.drop(df.columns[0], axis=1)

            # if key == 'bus':
            #     df = df.dropna(axis=1, how='all')
            # if key == 'gen':
            #     df.fillna(0, inplace=True)
            # len_cols = min(len(df.columns), len(header))
            # df = df.iloc[:, :len_cols]
            # df.columns = header
            mpc[key] = df

        elif key in ['gmd_bus', 'gmd_branch']:
            # line_number = string.count('\n', 0, string.index(attr))
            # if line_number > 0:
            #     header_line = lines[line_number - 1]
            #     header = header_line.strip().split("%")[-1].strip().split()
            # the keys with customized GMD data
            pattern = rf'mpc\.{key}\s*=\s*[\n]?(?P<data>.*?)[\n]?;'
            match = re.search(pattern, string, re.DOTALL)

            data = StringIO(match.groupdict()['data'][2:-2])
            # REVIEW: with multiple separators
            # df = pd.read_csv(data, sep="\\s+", header=None, engine="python")
            df = pd.read_csv(data, sep=r"\s+(?=(?:[^']*'[^']*')*[^']*$)", header=None, comment="%", engine="python")
            if df.shape[1] == len(HEADERS[key]):
                df.columns = HEADERS[key]
            elif df.shape[1] < len(HEADERS[key]):
                # fill the missing columns with 0
                df.columns = HEADERS[key][:df.shape[1]]
                for col in HEADERS[key][df.shape[1]:]:
                    df[col] = 0
            # drop the nan caused by tab splitter
            # df = df.drop(columns=[0])
            # len_cols = min(len(df.columns), len(header))
            # df = df.iloc[:, :len_cols]
            # df.columns = header
            mpc[key] = df
        elif key in ['branch_thermal']:
            # TODO: thermal problem is not considered
            pass

        elif key in ['gencost']:
            # TODO: update the loss function based on model type from gencost
            pass

        elif key in ['thermal_cap_x0', 'thermal_cap_y0']:
            # the keys with single column
            pattern = rf'mpc\.{key}\s*=\s*\[[\n]?(?P<data>.*?)[\n]?\];'
            match = re.search(pattern, string, re.DOTALL)

            mpc[key] = list(map(float, match.groupdict()['data'].split()))

        elif key in ['bus_sourceid', 'gen_sourceid', 'branch_sourceid']:
            # the keys with nodes and edges
            # NOTE: the sourceID is not matched the bus id.
            pass
            # pattern = rf'mpc\.{key}\s*=\s*\[[\n]?(?P<data>.*?);[\n]?\];'
            # match = re.search(pattern, string, re.DOTALL)

            # data = StringIO(match.groupdict()['data'])
            # df = pd.read_csv(data, sep="\t", header=None)
            # # drop the nan caused by tab splitter
            # df = df.drop(columns=[0])
            # mpc[key] = df.to_numpy()
    return mpc


def preprocess_mpc(mpc):
    r""" Preprocess the MPC data.
    The standard data format is defined in [1].
    The extended data format is defined in [2].

    Args:
        mpc (dict): MPC data.

    Returns:
        dict: Preprocessed MPC data.

    References:
        [1]: [MATPOWER manual](https://matpower.org/docs/MATPOWER-manual.pdf), Appendix B.
        [2]: [PowerModelsGMD.jl](https://github.com/lanl-ansi/PowerModelsGMD.jl/blob/master/README.md)
    """

    # convert to float
    for key in ['baseMVA', 'time_elapsed']:
        if key in mpc:
            mpc[key] = float(mpc[key])

    ''' process mpc['bus'] '''
    # convert `type` to one-hot encoder
    _series = mpc['bus'].type.astype("category").cat.set_categories(range(1, 5))
    bus_type_encoder = pd.get_dummies(_series).add_prefix('type_')
    mpc['bus'] = pd.concat([mpc['bus'], bus_type_encoder], axis=1)
    # drop type column
    mpc['bus'] = mpc['bus'].drop(['type'], axis=1)
    # build a dict to map bus_i to index: (key: bus_i, value: index)
    bus_id_idx = {bus_i: idx for idx, bus_i in enumerate(mpc['bus'].bus_i)}

    ''' process mpc['gen'] '''
    # replace `bus` with `bus_idx`
    mpc['gen'].bus = mpc['gen'].bus.replace(bus_id_idx)

    ''' process mpc['branch'] '''
    # replace `fbus` and `tbus` with `bus_idx`
    if "fbus" in mpc['branch'].columns:
        mpc['branch'].fbus = mpc['branch'].fbus.replace(bus_id_idx)
        mpc['branch'].tbus = mpc['branch'].tbus.replace(bus_id_idx)
    # NOTE: ACTIVSg cases has no f_bus and t_bus
    if "f_bus" in mpc['branch'].columns:
        mpc['branch'].f_bus = mpc['branch'].f_bus.replace(bus_id_idx)
        mpc['branch'].t_bus = mpc['branch'].t_bus.replace(bus_id_idx)
        mpc['branch'] = mpc['branch'].rename(columns={'f_bus': 'fbus', 't_bus': 'tbus'})
    # build a dict to map `branch_i` (not in the table) to index: (key: branch_i, value: index)
    branch_id_idx = {idx + 1: idx for idx, branch_i in enumerate(mpc['branch'].index)}

    ''' process mpc['gmd_bus'] '''
    # replace `parent_index` with `bus_idx`
    mpc['gmd_bus'].parent_index = mpc['gmd_bus'].parent_index.replace(bus_id_idx)
    # build a dict to map `gmd_bus_i` (not in the table) to index: (key: gmd_bus_i, value: index)
    gmd_bus_id_idx = {idx + 1: idx for idx, v in enumerate(mpc['gmd_bus'].index)}

    ''' process mpc['gmd_branch'] '''
    # replace `f_bus` and `t_bus` with `gmd_bus_idx`
    mpc['gmd_branch'].f_bus = mpc['gmd_branch'].f_bus.replace(gmd_bus_id_idx)
    mpc['gmd_branch'].t_bus = mpc['gmd_branch'].t_bus.replace(gmd_bus_id_idx)
    # replace `parent_index` with `branch_idx`
    mpc['gmd_branch'].parent_index = mpc['gmd_branch'].parent_index.replace(branch_id_idx)

    ''' process mpc['branch_gmd'] '''
    # NOTE: ACTIVSg cases has no branch_gmd
    if "branch_gmd" in mpc:
        # replace `hi_bus` and `lo_bus` with `bus_idx`
        mpc['branch_gmd'].hi_bus = mpc['branch_gmd'].hi_bus.replace(bus_id_idx)
        mpc['branch_gmd'].lo_bus = mpc['branch_gmd'].lo_bus.replace(bus_id_idx)
        # convert `type` to one-hot encoder
        branch_gmd_type = {"'xfmr'": 0,
                           "'transformer'": 0,
                           "'line'": 1,
                           "'series_cap'": 2}

        mpc['branch_gmd'].type = mpc['branch_gmd'].type.replace(branch_gmd_type)
        _series = mpc['branch_gmd'].type.astype("category").cat.set_categories(range(3))
        branch_gmd_type_encoder = pd.get_dummies(_series).add_prefix('type_')
        mpc['branch_gmd'] = pd.concat([mpc['branch_gmd'], branch_gmd_type_encoder], axis=1)
        # drop `type` column
        mpc['branch_gmd'] = mpc['branch_gmd'].drop(['type'], axis=1)
        # convert `config` to one-hot encoder
        branch_gmd_config = {"'none'": 0,
                             "'delta-delta'": 1,
                             "'delta-wye'": 2,
                             "'wye-delta'": 3,
                             "'wye-wye'": 4,
                             "'delta-gwye'": 5,
                             "'gwye-delta'": 6,
                             "'gwye-gwye'": 7,
                             "'gwye-gwye-auto'": 8}
        mpc['branch_gmd'].config = mpc['branch_gmd'].config.replace(branch_gmd_config)
        _series = mpc['branch_gmd'].config.astype("category").cat.set_categories(range(9))
        branch_gmd_config_encoder = pd.get_dummies(_series).add_prefix('config_')
        mpc['branch_gmd'] = pd.concat([mpc['branch_gmd'], branch_gmd_config_encoder], axis=1)
        # drop `config` column
        mpc['branch_gmd'] = mpc['branch_gmd'].drop(['config'], axis=1)

    return mpc


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


def process_args():
    """ Process args of inputs

    Returns:
        dict: Parsed arguments.
    """
    import argparse
    ACTS = ["relu", "rrelu", "hardtanh", "relu6", "sigmoid", "hardsigmoid", "tanh", "silu",
            "mish", "hardswish", "elu", "celu", "selu", "glu", "gelu", "hardshrink",
            "leakyrelu", "logsigmoid", "softplus", "tanhshrink"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", type=str, default=["epri21"], nargs='+',
                        help="list of names of networks, seperated by space")
    parser.add_argument("--force", action="store_true",
                        help="Force to reprocess data")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="weight decay rate for Adam")
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="hidden dimension in HGT")
    parser.add_argument("--num_heads", type=int, default=2,
                        help="number of heads in HGT")
    parser.add_argument("--num_conv_layers", type=int, default=4,
                        help="number of layers in HGT")
    parser.add_argument("--num_mlp_layers", type=int, default=4,
                        help="number of layers in MLP")
    parser.add_argument("--act", type=str, default="relu", choices=ACTS,
                        help="specify the activation function used")
    parser.add_argument("--conv_type", type=str, default="hgt", choices=["hgt", "han"],
                        help="select the type of convolutional layer (hgt or han)")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout rate")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of epochs in training")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="batch size in training")
    parser.add_argument("--seed", type=int, default=-1,
                        help="Random seed. Set `-1` to ignore random seed")
    parser.add_argument("--no_norm", action="store_true",
                        help="No normalization of the data")
    parser.add_argument("--test_split", type=float, default=0.2,
                        help="the proportion of datasets to use for testing")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--weight", action="store_true",
                        help="use weighted loss.")
    parser.add_argument("--log", action="store_true",
                        help="logging the training process")
    parser.add_argument("--verbose", action="store_true",
                        help="print the training process")
    parser.add_argument("--setting", type=str, default="gic", choices=["mld", "gic"],
                        help="Specify the problem setting, either `mld` or `gic`")
    args = vars(parser.parse_args())

    return args


def get_device(gpu_id=-1):
    r""" Get the device where the model is running on.

    Args:
        gpu_id (int): GPU ID to use. Set -1 to use CPU.

    Returns:
        torch.device: Device where the model is running on.
    """
    import torch
    if gpu_id < 0 or (not torch.cuda.is_available()) or (gpu_id >= torch.cuda.device_count()):
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{gpu_id}")
    return device


def export_gmd_blocker_matpower(labels, in_fn, out_fn):
    """ Export the GMD_BLOCKER labels to the matpower file.

    Args:
        labels (np.array): Labels of the gmd_blocker.
        in_fn (str): Input matpower file.
        out_fn (str): Output matpower file.
    """
    gic_str = """\n%% gmd_blocker data\n%column_names% gmd_bus status\nmpc.gmd_blocker = {\n"""
    for idx, v in enumerate(labels):
        gic_str += f"\t{idx+1}\t{v.item()}\n"
    gic_str += "};"

    with open(out_fn, "w") as f1:
        with open(in_fn, "r") as f2:
            f1.write(f2.read() + gic_str)
