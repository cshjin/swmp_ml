import os
import re
from io import StringIO

import pandas as pd

HEADERS = {
    "gen": ['bus', 'Pg', 'Qg', 'Qmax', 'Qmin',
            'Vg', 'mBase', 'status', 'Pmax', 'Pmin',
            'Pc1', 'Pc2', 'Qc1min', 'Qc1max',
            'Qc2min', 'Qc2max', 'ramp_agc', 'ramp_10', 'ramp_30', 'ramp_q', 'apf'],
    "bus": ['bus_i', 'type', 'Pd', 'Qd', 'Gs', 'Bs', 'area',
            'Vm', 'Va', 'baseKV', 'zone', 'Vmax', 'Vmin'],
    "gencost": ['model', 'startup', 'shutdown', 'ncost', 'c0', 'c1', 'c2'],
    "branch": ['fbus', 'tbus', 'r', 'x', 'b',
               'rateA', 'rateB', 'rateC', 'ratio',
               'angle', 'status', 'angmin', 'angmax'],
    "gmd_bus": ['parent_index', 'status', 'g_gnd', 'name'],
    "gmd_branch": ['f_bus', 't_bus', 'parent_index', 'br_status',
                   'br_r', 'br_v', 'len_km', 'name'],
    "branch_gmd": ['hi_bus', 'lo_bus', 'gmd_br_hi', 'gmd_br_lo',
                   'gmd_k', 'gmd_br_series', 'gmd_br_common', 'baseMVA', 'type', 'config'],
    "branch_thermal": ['xfmr', 'temperature_ambient', 'hotspot_instant_limit',
                       'hotspot_avg_limit', 'hotspot_rated', 'topoil_time_const', 'topoil_rated',
                       'topoil_init', 'topoil_initialized', 'hotspot_coeff'],
    "bus_gmd": ['lat', 'lon'],
}


def read_mpc(fn):
    r""" Read the MPC MATPOWER file ".m" into dictionary.
    Args:
        fn (str): Filename.

    Returns:
        dict: A dictionary with keys of attributes.

    References:
        [1]: MATPOWER manual (https://matpower.org/docs/MATPOWER-manual.pdf), Appendix B.
    """
    mpc = {}
    with open(fn, "r") as f:
        string = f.read()

    # find match with `mpc.***`
    matches = re.findall(r"mpc\.\w+", string)
    for attr in matches:
        key = attr.split(".")[1]

        # process with different patterns
        if key in ['version', 'baseMVA', 'time_elapsed']:
            # the key with only one value
            pattern = rf'mpc\.{key}\s*=\s*(?P<data>.*?);'
            match = re.search(pattern, string, re.DOTALL)
            value = match.groupdict()['data'].strip("'").strip('"')
            # if key == 'baseMVA' or key == 'time_elapsed':
            #     value = float(value)
            mpc[key] = value

        elif key in ['gen', 'gencost', 'bus', 'branch']:
            # the keys with standard MATPOWER data
            pattern = rf'mpc\.{key}\s*=\s*\[[\n]?(?P<data>.*?)[\n]?\];'
            match = re.search(pattern, string, re.DOTALL)
            # convert to string for pandas dataframe
            data = StringIO(match.groupdict()['data'])
            df = pd.read_csv(data, sep="\t", header=None)
            # drop the nan caused by tab splitter
            df = df.drop(columns=[0])
            df.columns = HEADERS[key]
            mpc[key] = df

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

        elif key in ['gmd_bus', 'gmd_branch', 'branch_gmd', 'branch_thermal', 'bus_gmd']:
            # the keys with customized GMD data
            pattern = r'mpc\.{}\s*=\s*[\n]?(?P<data>.*?)[\n]?;'.format(key)
            match = re.search(pattern, string, re.DOTALL)

            data = StringIO(match.groupdict()['data'][2:-2])
            # REVIEW: with multiple separators
            df = pd.read_csv(data, sep="\t", header=None, engine="python")
            # drop the nan caused by tab splitter
            df = df.drop(columns=[0])
            df.columns = HEADERS[key]
            mpc[key] = df
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
    parser.add_argument("--num_conv_layers", type=int, default=1,
                        help="number of layers in HGT")
    parser.add_argument("--num_mlp_layers", type=int, default=4,
                        help="number of layers in MLP")
    parser.add_argument("--activation", type=str, default="relu", choices=ACTS,
                        help="specify the activation function used")
    parser.add_argument("--conv_type", type=str, default="hgt", choices=["hgt", "han"],
                        help="select the type of convolutional layer (hgt or han)")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout rate")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of epochs in training")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="batch size in training")
    parser.add_argument("--normalize", action="store_true",
                        help="normalize the data")
    parser.add_argument("--test_split", type=float, default=0.2,
                        help="the proportion of datasets to use for testing")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--weight", action="store_true",
                        help="use weighted loss.")
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
