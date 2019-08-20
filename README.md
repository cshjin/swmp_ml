# PowerModelsGMD.jl

<!--
Release: 
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://lanl-ansi.github.io/PowerModelsGMD.jl/stable/)
-->

Dev:
[![Build Status](https://travis-ci.org/lanl-ansi/PowerModelsGMD.jl.svg?branch=master)](https://travis-ci.org/lanl-ansi/PowerModelsGMD.jl)
<!--
[![codecov](https://codecov.io/gh/lanl-ansi/PowerModelsGMD.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/lanl-ansi/PowerModelsGMD.jl)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://lanl-ansi.github.io/PowerModelsGMD.jl/latest/)
</p>
-->

PowerModelsGMD.jl provides extensions to [PowerModels.jl](https://github.com/lanl-ansi/PowerModels.jl) to evaluate the risks posed by solar storms, to analyze and mitigate the potential effects of Geomagnetic Disturbances (GMDs) on the power grid.
This open-source toolbox is a greatly accessible, high-performance and easy-to-handle alternative to other commercially available software solutions.



## Core Problem Specifications

* Geomagnetically Induced Current (GIC) DC Solve: Solve for steady-state dc currents on lines resulting from induced dc voltages on lines
* Coupled GIC + AC Optimal Power Flow (OPF): Solve the AC-OPF problem for a network subjected to GIC. The dc network couples to the ac network by means of reactive power loss in transformers.
* Coupled GIC + AC Minimum Load Shed (MLS). Solve the minimum-load shedding problem for a network subjected to GIC.
* Coupled GIC + AC Optimal Transmission Switching (OTS). Solve the minimum-load shedding problem for a network subjected to GIC where lines and transformers can be opened or closed.



## Installation

First, follow the installation instructions for [PowerModels.jl](https://github.com/lanl-ansi/PowerModels.jl).
From the Julia package manager REPL type
```
add https://github.com/lanl-ansi/PowerModelsGMD.jl.git
```

Test with,
```
test PowerModelsGMD
```



## Quick Start

The most common use case is a quasi-dc solve followed by an AC-OPF where the currents from the quasi-dc solve are constant parameters that determine the reactive power consumption of transformers on the system.

```
using PowerModels; using PowerModelsGMD; using Ipopt
network_file = joinpath(dirname(pathof(PowerModelsGMD)), "../test/data/epri21.m")
case = PowerModels.parse_file(network_file)

solver = with_optimizer(Ipopt.Optimizer)
result = PowerModelsGMD.run_ac_gmd_opf_decoupled(case, solver)
```



## Function Reference
<!-- 
1) check that the test datasets correspond to those used in the test cases
2) review and poitentially rework descriptions
-->

### GIC

This solves for the quasi-dc voltage and currents on a system
`run_gmd("test/data/b4gic.m", solver)`

For large systems of greater than 10,000 buses consider using the Lehtinen-Pirjola (LP) form which uses a matrix solve instead of an optimizer.
This is called by omitting the solver parameter
`run_gmd("test/data/b4gic.m")`

To save branch currents in addition to bus voltages
```
setting = Dict{String,Any}("output" => Dict{String,Any}("branch_flows" => true))
run_gmd("test/data/b4gic.m", solver, setting=setting)
```


### GIC -> AC-OPF

This solves for the quasi-dc voltages and currents, and uses the calculated quasi-dc currents through trasformer windings as inputs to a an AC-OPF to calculate the increase in transformer reactive power consumption.
`run_ac_gmd_opf_decoupled("test/data/b4gic.m")`


### GIC + AC-OPF

This solves the quasi-dc voltages and currents and the AC-OPF concurrently. This formulation has limitations in that it does not model increase in transformer reactive power consumption resulting from changes in the ac terminal voltages. 
Additionally, it may report higher reactive power consumption than reality on account of relaxing the "effective" transformer quasi-dc winding current magnitude.
`run_ac_gmd_opf("test/data/b4gic.m")`


### GIC + AC-MLS

Solve the minimum-load shedding problem for a network subjected to GIC with fixed topology.
`run_ac_gmd_ml("test/data/case24_ieee_rts_0.m")`


### GIC + AC-OTS

 Solve the minimum-load shedding problem for a network subjected to GIC where lines and transformers can be opened or closed.
`run_ac_gmd_ots("test/data/ots_test.m")`



## Data Reference

PowerModelsGMD.jl uses several extensions to the PowerModels.jl data format.
For generality, it uses a separate dc network defined by the `gmd_bus` and `gmd_branch` tables.
To correctly calculate the increased reactive power consumption of each transformer, the `branch_gmd` table adds all winding configuration related data. Furthermore, `branch_thermal` table adds thermal data necessary to determine the temperature changes in transformers.
If the results are plotted spatially, for convinience, `bus_gmd` table includes the latitude and longitude of buses in the ac network.


### GMD Bus Data Table

This table includes
* `parent_index` - the index of the corresponding bus in the ac network 
* `status` - binary value that sets the status of the bus (1 is enabled, 0 is disabled)
* `g_gnd` - the admittance to ground in unit of Siemens
* `name` - a descriptive name for the bus

```
%column_names% parent_index status g_gnd name
mpc.gmd_bus = {
	1	1	5	'dc_sub1'
	2	1	5	'dc_sub2'
	1	1	0	'dc_bus1'
	2	1	0	'dc_bus2'
	3	1	0	'dc_bus3'
	4	1	0	'dc_bus4'
};
```


### GMD Branch Data Table

This table includes
* `f_bus` - the "from" bus in the gmd bus table
* `t_bus` - to "to" bus in the gmd bus table
* `parent_index` - the index of the corresponding branch in the ac network
* `br_status` - binary value that sets the statusof the branch (1 is enabled, 0 is disabled)
* `br_r` - the branch resistance (in unit of Ohms)
* `br_v` - the induced quasi-dc voltage in volts
* `len_km` - the length of the branch (in unit of km) -- not required
* `name` - a descriptive name for the branch

```
%column_names% f_bus t_bus parent_index br_status br_r br_v len_km name
mpc.gmd_branch = {
	3	1	1	1	0.1	0	0	'dc_xf1_hi'
	3	4	2	1	1.00073475	170.78806587354	170.78806587354	'dc_br1'
	4	2	3	1	0.1	0	0	'dc_xf2_hi'
};
```


### Branch GMD Data Table

This table includes
* `hi_bus` - the index of the high-side bus (in the ac network)
* `lo_bus` - the index of the low-side bus (in the ac network)
* `gmd_br_hi` - the index of the gmd branch corresponding to the high-side winding (for two-winding transformers)
* `gmd_br_lo` - the index of the gmd branch corresponding to the low-side winding (for two-winding transformers)
* `gmd_k` - the scaling factor used to calculate reactive power consumption in per-unit as a function of effective winding current (in per-unit)
* `gmd_br_series` - the index of the gmd branch corresponding to the series winding (for autotransformers)
* `gmd_br_common` - the index of the gmd branch corresponding to the common winding (for autotransformers) 
* `baseMVA` - the MVA base of the trasformer
* `type` - type of the branch -- "xf" / "transformer, or "line" 
* `config` - the winding configuration of the transformer -- currently "gwye-gwye", "gwye-delta", "delta-delta", and "gwye-gwye-auto" are supported

```
%column_names% hi_bus lo_bus gmd_br_hi gmd_br_lo gmd_k gmd_br_series gmd_br_common baseMVA type config
mpc.branch_gmd = {
	1	3	1	-1	1.793	-1	-1	100	'xf'	'gwye-delta'
	1	2	-1	-1	0	-1	-1	100	'line'	'none'
	2	4	3	-1	1.793	-1	-1	100	'xf'	'gwye-delta'
};
```


### Branch Thermal Data Table 

This table includes
* `xfmr` - binary value that defines if the branch is a transformer (1 is transformer, 0 is not a transformer)
* `temperature_ambient` - ambient temperature of the transformer (in unit of Celsius)
* `hotspot_instant_limit` - 1-hour hotspot temperature limit of the transformer (in unit of Celsius)
* `hotspot_avg_limit` - 8-hour hotspot temperature limit of the transformer (in unit of Celsius)
* `hotspot_rated` - ...
* `topoil_time_const` - top-oil temperature-rise time-constant (in unit of minutes)
* `topoil_rated` - top-oil temperature-rise of the transformer at rated power (in unit of Celsius)
* `topoil_init` - initial top-oil temperature of the transformer (in unit of Celsius)
* `topoil_initialized` - binary value that defines the initial top-oil temperature of the transformer (1 is initial top-oil temperature starts with `topoil_init` value, 0 is initial top-oil temperature starts with steady-state value)
* `hotspot_coeff` - relationship of hotspot temperature rise to Ieff (in unit of Celsius/amp)

```
%column_names% xfmr temperature_ambient hotspot_instant_limit hotspot_avg_limit hotspot_rated topoil_time_const topoil_rated topoil_init topoil_initialized hotspot_coeff
mpc.branch_thermal = {
	1	25	280	240	150	71	75	0	1	0.63
	0	25	-1	-1	-1	-1	-1	-1	-1	-1
	1	25	280	240	150	71	75	0	1	0.63
};
```


### Bus GMD Data Table

This table includes 
* `lat` - latitude coordinate of bus in the ac network
* `lon` - longitude coordinate of bus in the ac network

```
%column_names% lat lon
mpc.bus_gmd = {
	40	-89
	40	-87
	40	-89
	40	-87
};
```



<!-- 
## Development



## Acknowledgments

This code has been developed as part of the Advanced Network Science Initiative at Los Alamos National Laboratory.



## Citing PowerModelsGMD.jl

If you find PowerModelsGMD.jl useful in your work, we kindly request that you cite the following publication:

-->



## License

This code is provided under a BSD license as part of the Multi-Infrastructure Control and Optimization Toolkit (MICOT) project, LA-CC-13-108.


