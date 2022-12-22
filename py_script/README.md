# README

## Package Installation

`sh setup.sh cpu` if you have only cpu available, otherwise
`sh setup.sh gpu`.

## Data Description

### Customized MATPOWER datafile "*.m" 

* keys
  | key                 | type   | description                     |
  | ------------------- | ------ | ------------------------------- |
  | mpc.version         | string |                                 |
  | mpc.baseMVA         | int    |                                 |
  | mpc.bus             | table  |                                 |
  | mpc.gen             | table  |                                 |
  | mpc.branch          | table  |                                 |
  | mpc.gencost         | table  |                                 |
  | mpc.gmd_bus         | table  |                                 |
  | mpc.gmd_branch      | table  |                                 |
  | mpc.branch_gmd      | table  |                                 |
  | mpc.branch_thermal  | table  |                                 |
  | mpc.bus_gmd         | table  | geo location                    |
  | mpc.time_elapsed    | float  |                                 |
  | mpc.thermal_cap_x0  | list   | thermal caps [per unit]         |
  | mpc.thermal_cap_y0  | list   | thermal caps [percent per unit] |
  | mpc.bus_sourceid    | table* | SourceID data                   |
  | mpc.gen_sourceid    | table* | SourceID data                   |
  | mpc.branch_sourceid | table* | SourceID data                   |
  
### Standard keys in MATPOWER

* `x`: exclude from features
* `-`: read value from PowerModelGMD `results`
* `o`: convert to one-hot encoders

* `bus`
  | selected | column | type  | description               |
  | :------: | ------ | ----- | ------------------------- |
  |    x     | bus_i  | int   | bus number                |
  |  o (x)   | type   | int   | bus type (1-4)            |
  |          | Pd     | float | real power demand         |
  |          | Qd     | float | reactive power demand     |
  |          | Gs     | float | shunt conductance         |
  |          | Bs     | float | shunt susceptance         |
  |    x     | area   | int   | area number               |
  |    -     | Vm     | float | voltage magnitude         |
  |    -     | Va     | float | voltage angle             |
  |          | baseKV | float | base voltage              |
  |    x     | zone   | int   | loss zone                 |
  |    x     | Vmax   | float | maximum voltage magnitude |
  |    x     | Vmin   | float | minimum voltage magnitude |
  
* `gen`
  | selected | column   | type  | description                                    |
  | :------: | -------- | ----- | ---------------------------------------------- |
  |    x     | bus      | int   | bus number                                     |
  |    x     | Pg       | float | real power output                              |
  |    x     | Qg       | float | reactive power output                          |
  |          | Qmax     | float | maximum reactive power output                  |
  |          | Qmin     | float | minimum reactive power output                  |
  |          | Vg       | float | voltage magnitude setpoint                     |
  |    x     | mBase    | float | total MVA base of machine                      |
  |    x     | status   | float | status, > 0, in-service, <= 0 out-of-service   |
  |          | Pmax     | float | maximum real power output                      |
  |          | Pmin     | float | minimum real power output                      |
  |    x     | Pc1      | float | lower real power output of PQ capability curve |
  |    x     | Pc2      | float | upper real power output of PQ capability curve |
  |    x     | Qc1min   | float | minimum reactive power output at PC1           |
  |    x     | Qc1max   | float | maximum reactive power output at PC1           |
  |    x     | Qc2min   | float | minimum reactive power output at PC2           |
  |    x     | Qc2max   | float | maximum reactive power output at PC2           |
  |    x     | ramp_agc | float | ramp rate for load following/AGC               |
  |    x     | ramp_10  | float | ramp rate for 10 minute reserves               |
  |    x     | ramp_30  | float | ramp rate for 30 minute reserves               |
  |    x     | ramp_q   | float | ramp rate for reactive power (2 sec timescale) |
  |    x     | apf      | float | area participation factor                      |

* `branch`
  | selected | column | type   | description                                              |
  | :------: | ------ | ------ | -------------------------------------------------------- |
  |    x     | fbus   | int    | "from" bus number                                        |
  |    x     | tbus   | int    | "to" bus number                                          |
  |          | r      | float  | resistance                                               |
  |          | x      | float  | reactance                                                |
  |          | b      | float  | total line charging susceptance                          |
  |          | rateA  | float  | MVA rating A (long term rating), set to 0 for unlimited  |
  |    x     | rateB  | float  | MVA rating B (short term rating), set to 0 for unlimited |
  |    x     | rateC  | float  | MVA rating C (emergency rating), set to 0 for unlimited  |
  |          | ratio  | float  | TAP: transformer off nominal turns ratio                 |
  |          | angle  | float  | SHIFT:  transformer phase shift angle (degrees)          |
  |    x     | status | binary | initial branch status, 1: in-service, 0: out-of-service  |
  |          | angmin | float  | minimum angle difference                                 |
  |          | angmax | float  | maximum angle difference                                 |

* `gencost` (ignore)
  | selected | column   | type  | description                                                     |
  | :------: | -------- | ----- | --------------------------------------------------------------- |
  |          | model    | int   | cost model, 1: linear, 2: polynomial                            |
  |          | startup  | float | startup cost                                                    |
  |          | shutdown | float | shutdown cost                                                   |
  |          | ncost    | int   | number N = n + 1 of data points defining an n-segment piecewise |
  |          | c0       | float | coef                                                            |
  |          | c1       | float | coef                                                            |
  |          | c2       | float | coef                                                            |

  * `gencost` looks like associated with `gen`, but miss the `gen_i` or `bus_i`
  * c0, c1, c2 are not exact with 3 coefficients, may be less or more, depend on `ncost`
  * c0, c1, c2 are the coeffients of polynomial function, starting from high to low

### Customized keys in MATPOWER

* `gmd_bus`
  | selected | columns      | dtype  | description                                                           |
  | :------: | ------------ | ------ | --------------------------------------------------------------------- |
  |    x     | parent_index | int    | index of corresponding ac network bus                                 |
  |    x     | status       | binary | binary value that defines the status of bus (1: enabled, 0: disabled) |
  |          | g_gnd        | float  | admittance to ground (in unit of Siemens)                             |
  |    x     | name         | string | a descriptive name for the bus                                        |

* `gmd_branch`
  | selected | columns      | dtype  | description                                                              |
  | :------: | ------------ | ------ | ------------------------------------------------------------------------ |
  |    x     | f_bus        | int    | "from" bus in the gmd bus table                                          |
  |    x     | t_bus        | int    | "to" bus in the gmd bus table                                            |
  |    x     | parent_index | int    | index of corresponding ac network branch                                 |
  |    x     | br_status    | binary | binary value that defines the status of branch (1: enabled, 0: disabled) |
  |          | br_r         | float  | branch resistance (in unit of Ohms)                                      |
  |          | br_v         | float  | induced quasi-dc voltage (in unit of Volts)                              |
  |    x     | len_km       | float  | length of branch (in unit of km)   optional                              |
  |    x     | name         | string | a descriptive name for the branch                                        |

* `branch_gmd`
  | selected | columns       | dtype  | description                                                                                                                                                                       |
  | :------: | ------------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  |    x     | hi_bus        | int    | index of high-side ac network bus                                                                                                                                                 |
  |    x     | lo_bus        | int    | index of low-side ac network bus                                                                                                                                                  |
  |    x     | gmd_br_hi     | int    | index of gmd branch corresponding to high-side winding (for two-winding transformers)                                                                                             |
  |    x     | gmd_br_lo     | int    | index of gmd branch corresponding to low-side winding (for two-winding transformers)                                                                                              |
  |          | gmd_k         | float  | scaling factor to calculate reactive power consumption as a function of effective winding current (in per-unit)                                                                   |
  |    x     | gmd_br_series | int    | index of gmd branch corresponding to series winding (for auto-transformers)                                                                                                       |
  |    x     | gmd_br_common | int    | index of gmd branch corresponding to common winding (for auto-transformers)                                                                                                       |
  |          | baseMVA       | float  | MVA base of transformer                                                                                                                                                           |
  |    o     | type          | string | type of branch -- "xfmr" / "transformer", "line", or "series_cap"                                                                                                                 |
  |    o     | config        | string | winding configuration of transformer -- currently "delta-delta", "delta-wye", "wye-delta", "wye-wye", "delta-gwye", "gwye-delta", "gwye-gwye", and "gwye-gwye-auto" are supported |

* `branch_thermal` (ignore)
  | selected | columns               | dtype  | description                                                                                                                                                            |
  | :------: | --------------------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  |          | xfmr                  | binary | binary value that defines if branch is a transformer (1: transformer, 0: not a transformer)                                                                            |
  |          | temperature_ambient   | float  | ambient temperature of transformer (in unit of Celsius)                                                                                                                |
  |          | hotspot_instant_limit | float  | 1-hour hot-spot temperature limit of transformer (in unit of Celsius)                                                                                                  |
  |          | hotspot_avg_limit     | float  | 8-hour hot-spot temperature limit of transformer (in unit of Celsius)                                                                                                  |
  |          | hotspot_rated         | float  | hot-spot temperature-rise of transformer at rated power (in unit of Celsius)                                                                                           |
  |          | topoil_time_const     | float  | top-oil temperature-rise time-constant of transformer (in unit of minutes)                                                                                             |
  |          | topoil_rated          | float  | top-oil temperature-rise of transformer at rated power (in unit of Celsius)                                                                                            |
  |          | topoil_init           | float  | initial top-oil temperature of transformer (in unit of Celsius)                                                                                                        |
  |          | topoil_initialized    | binary | binary value that defines the initial top-oil temperature of transformer (1: temperature starts with topoil_init value, 0: temperature starts with steady-state value) |
  |          | hotspot_coeff         | float  | relationship of hot-spot temperature rise to Ieff (in unit of Celsius/Amp)                                                                                             |

* `bus_gmd` (ignore, REVIEW)
  | selected | columns | dtype | description                                                             |
  | :------: | ------- | ----- | ----------------------------------------------------------------------- |
  |          | lat     | float | latitude coordinate of ac network bus and corresponding dc network bus  |
  |          | lon     | float | longitude coordinate of ac network bus and corresponding dc network bus |

* `time_elapsed` (ignore)
  | selected | columns | dtype | description  |
  | :------: | ------- | ----- | ------------ |
  |          | seconds | float | time elapsed |

<!-- REVIEW: meaning of the columns? -->
* `thermal_cap_x0`
  | selected | columns | dtype | description |
  | :------: | ------- | ----- | ----------- |
  |          | A       | float | per unit    |
  |          | B       | float |             |
  |          | C       | float |             |
  |          | D       | float |             |
  |          | E       | float |             |
  |          | F       | float |             |
  |          | G       | float |             |
  |          | H       | float |             |
  |          | I       | float |             |
  |          | J       | float |             |
  |          | K       | float |             |

* `thermal_cap_y0`
  | selected | columns | dtype | description      |
  | :------: | ------- | ----- | ---------------- |
  |          | A       | float | percent per unit |
  |          | B       | float |                  |
  |          | C       | float |                  |
  |          | D       | float |                  |
  |          | E       | float |                  |
  |          | F       | float |                  |
  |          | G       | float |                  |
  |          | H       | float |                  |
  |          | I       | float |                  |
  |          | J       | float |                  |
  |          | K       | float |                  |

* `bus_sourceid`
  | selected | columns | dtype  | description                                      |
  | :------: | ------- | ------ | ------------------------------------------------ |
  |          | bus_sid | string | bus source id (in string format ending with " ") |

* `gen_sourceid`
  | selected | columns | dtype  | description                                            |
  | :------: | ------- | ------ | ------------------------------------------------------ |
  |          | gen_sid | string | generator source id (in string format ending with " ") |

* `branch_sourceid`
  | selected | columns    | dtype  | description                                         |
  | :------: | ---------- | ------ | --------------------------------------------------- |
  |          | fbus       | int    | from bus source id                                  |
  |          | tbus       | int    | target bus source id                                |
  |          | branch_sid | string | branch source id (in string format ending with " ") |



  `gmd_vdc`: output from optimizer
    dc value after the gic power flow.



## running the code

<!-- clf -->
```
python test_dataset_v2.py  --force
```

<!-- reg -->
```
python test_dataset_v2.py  --force --problem reg --epochs 1000 --lr 0.01
```