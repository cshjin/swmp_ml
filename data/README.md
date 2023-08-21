# Data Folder Description

In the `data` folder, we have the following subfolders:

```bash
data/
├── excel
├── gic
│   ├── results
│   │   ├── epri21
│   │   └── uiuc150
│   └── src
└── matpower
```

* `excel` is the folder for `EXCEL` data format. __Note__: current heuristic solvers only support `EXCEL` data format.
* `gic` is the folder for `GIC` problem, including the `results` from heuristic solvers and `src` code for heuristic solvers.
* `matpower` is the folder for `MATPOWER` data format. This is the folder for processing `PyG` dataset.

## Running the Heuristic Solvers

* Install the Julia environment by following the instructions in [Julia](https://julialang.org/downloads/). 
* Install the required packages by running the following command under the `data` folder:

```bash
julia install.jl
```

* Run the GIC blocker problem by running the following command under the `data/gic` folder:

```bash
julia gic_opf_blockers.jl
```

It will create new result file under the `data/gic/results` folder.

Usage of the `gic_opf_blockers.jl` is as follows:

```bash
usage: gic_opf_blockers.jl [--network NETWORK] [--model MODEL]
                        [--optimizer OPTIMIZER]
                        [--time_limit TIME_LIMIT]
                        [--efield_mag EFIELD_MAG]
                        [--efield_dir EFIELD_DIR]
                        [--tot_num_blockers TOT_NUM_BLOCKERS]
                        [--output_dir_name OUTPUT_DIR_NAME]
                        [--run_id RUN_ID] [-h]

optional arguments:
  --network NETWORK      (default: "epri21")
  --model MODEL          (default: "ac_polar")
  --optimizer OPTIMIZER
                        (default: "juniper")
  --time_limit TIME_LIMIT
                        (type: Float64, default: 3600.0)
  --efield_mag EFIELD_MAG
                        (type: Float64, default: 10.0)
  --efield_dir EFIELD_DIR
                        (type: Float64, default: 45.0)
  --tot_num_blockers TOT_NUM_BLOCKERS
                        (type: Int64, default: 10)
  --output_dir_name OUTPUT_DIR_NAME
                        (default: "./results/")
  --run_id RUN_ID       (type: Int64, default: 1)
  -h, --help            show this help message and exit
```

* generate multiple perturbations by running the following command under the `data/gic` folder:

```bash
# generate 200 perturbations
python gen_gic_data.py --nums 200

usage: gen_gic_data.py [-h] [--nums NUMS] [--network NETWORK] [--model MODEL] [--optimizer OPTIMIZER]

optional arguments:
  -h, --help            show this help message and exit
  --nums NUMS           Number of samples to generate
  --network NETWORK     Network name
  --model MODEL         Model type, options=[ac_polar, ac_rect, soc_polar, soc_rect]
  --optimizer OPTIMIZER
                        Optimizer type, options=[juniper, scip]
```

