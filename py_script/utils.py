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


def read_file(fn):
    """ Read the MPC file ".m" into dictionary.

    Args:
        fn (str): Filename.

    Returns:
        dict: A dictionary with keys of attributes.
    """
    mpc = {}
    with open(fn, "r") as f:
        string = f.read()

    matches = re.findall(r"mpc\.\w+", string)
    for attr in matches:
        key = attr.split(".")[1]
        # process with different patterns

        if key in ['version', 'baseMVA', 'time_elapsed']:
            # the key with only one value
            pattern = rf'mpc\.{key}\s*=\s*(?P<data>.*?);'
            match = re.search(pattern, string, re.DOTALL)

            mpc[key] = match.groupdict()['data'].strip("'").strip('"')

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
