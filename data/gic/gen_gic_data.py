""" Generate GIC data for the given network and model type in parallel."""
import argparse
import os

from joblib import Parallel, delayed

parser = argparse.ArgumentParser()
parser.add_argument("--nums", type=int, default=200,
                    help="Number of samples to generate")
parser.add_argument("--network", type=str, default="epri21",
                    help="Network name")
parser.add_argument("--model", type=str, default="ac_polar",
                    help="Model type, options=[ac_polar, ac_rect, soc_polar, soc_rect]")
parser.add_argument("--optimizer", type=str, default="stochastic",
                    help="Optimizer type, options=[stochastic, juniper, scip]")
parser.add_argument("--efield_mag", type=float, default=5.0,
                    help="Magnitude of the electric field")
parser.add_argument("--efield_dir", type=float, default=45.0,
                    help="Direction of the electric field")
args = parser.parse_args()
args = vars(args)


def func(iter):
    cmd = f"""julia gic_opf_blockers_stochastic.jl \\
    --network {args['network']} \\
    --model {args['model']} \\
    --optimizer {args['optimizer']} \\
    --efield_mag {args['efield_mag']} \\
    --efield_dir {args['efield_dir']} \\
    --run_id {iter} > /dev/null 2>&1"""
    # print(cmd)
    os.system(cmd)


# TODO: add perturbations to efield_mag and efield_dir  -- add efiled_mag and efield_dir to the command line
Parallel(n_jobs=-1, prefer="threads")(delayed(func)(iter) for iter in range(1, args["nums"] + 1))
