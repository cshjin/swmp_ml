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
  
  
### Customized keys in MATPOWER

* `gmd_bus`
  | columns      | dtype  | description                                                           |
  | ------------ | ------ | --------------------------------------------------------------------- |
  | parent_index | int    | index of corresponding ac network bus                                 |
  | status       | binary | binary value that defines the status of bus (1: enabled, 0: disabled) |
  | g_gnd        | float  | admittance to ground (in unit of Siemens)                             |
  | name         | string | a descriptive name for the bus                                        |

* `gmd_branch`
  | columns      | dtype  | description                                                              |
  | ------------ | ------ | ------------------------------------------------------------------------ |
  | f_bus        | int    | "from" bus in the gmd bus table                                          |
  | t_bus        | int    | "to" bus in the gmd bus table                                            |
  | parent_index | int    | index of corresponding ac network branch                                 |
  | br_status    | binary | binary value that defines the status of branch (1: enabled, 0: disabled) |
  | br_r         | float  | branch resistance (in unit of Ohms)                                      |
  | br_v         | float  | induced quasi-dc voltage (in unit of Volts)                              |
  | len_km       | float  | length of branch (in unit of km)   optional                              |
  | name         | string | a descriptive name for the branch                                        |

* `branch_gmd`
  | columns       | dtype  | description                                                                                                                                                                       |
  | ------------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | hi_bus        | int    | index of high-side ac network bus                                                                                                                                                 |
  | lo_bus        | int    | index of low-side ac network bus                                                                                                                                                  |
  | gmd_br_hi     | int    | index of gmd branch corresponding to high-side winding (for two-winding transformers)                                                                                             |
  | gmd_br_lo     | int    | index of gmd branch corresponding to low-side winding (for two-winding transformers)                                                                                              |
  | gmd_k         | float  | scaling factor to calculate reactive power consumption as a function of effective winding current (in per-unit)                                                                   |
  | gmd_br_series | int    | index of gmd branch corresponding to series winding (for auto-transformers)                                                                                                       |
  | gmd_br_common | int    | index of gmd branch corresponding to common winding (for auto-transformers)                                                                                                       |
  | baseMVA       | float  | MVA base of transformer                                                                                                                                                           |
  | type          | string | type of branch -- "xfmr" / "transformer", "line", or "series_cap"                                                                                                                 |
  | config        | string | winding configuration of transformer -- currently "delta-delta", "delta-wye", "wye-delta", "wye-wye", "delta-gwye", "gwye-delta", "gwye-gwye", and "gwye-gwye-auto" are supported |

* `branch_thermal`
  | columns               | dtype  | description                                                                                                                                                            |
  | --------------------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | xfmr                  | binary | binary value that defines if branch is a transformer (1: transformer, 0: not a transformer)                                                                            |
  | temperature_ambient   | float  | ambient temperature of transformer (in unit of Celsius)                                                                                                                |
  | hotspot_instant_limit | float  | 1-hour hot-spot temperature limit of transformer (in unit of Celsius)                                                                                                  |
  | hotspot_avg_limit     | float  | 8-hour hot-spot temperature limit of transformer (in unit of Celsius)                                                                                                  |
  | hotspot_rated         | float  | hot-spot temperature-rise of transformer at rated power (in unit of Celsius)                                                                                           |
  | topoil_time_const     | float  | top-oil temperature-rise time-constant of transformer (in unit of minutes)                                                                                             |
  | topoil_rated          | float  | top-oil temperature-rise of transformer at rated power (in unit of Celsius)                                                                                            |
  | topoil_init           | float  | initial top-oil temperature of transformer (in unit of Celsius)                                                                                                        |
  | topoil_initialized    | binary | binary value that defines the initial top-oil temperature of transformer (1: temperature starts with topoil_init value, 0: temperature starts with steady-state value) |
  | hotspot_coeff         | float  | relationship of hot-spot temperature rise to Ieff (in unit of Celsius/Amp)                                                                                             |

* `bus_gmd`
  | columns | dtype | description                                                             |
  | ------- | ----- | ----------------------------------------------------------------------- |
  | lat     | float | latitude coordinate of ac network bus and corresponding dc network bus  |
  | lon     | float | longitude coordinate of ac network bus and corresponding dc network bus |

* `time_elapsed`
  | columns | dtype | description  |
  | ------- | ----- | ------------ |
  | seconds | float | time elapsed |

<!-- REVIEW: meaning of the columns? -->
* `thermal_cap_x0`
  | columns | dtype | description |
  | ------- | ----- | ----------- |
  | A       | float | per unit    |
  | B       | float |             |
  | C       | float |             |
  | D       | float |             |
  | E       | float |             |
  | F       | float |             |
  | G       | float |             |
  | H       | float |             |
  | I       | float |             |
  | J       | float |             |
  | K       | float |             |

* `thermal_cap_y0`
  | columns | dtype | description      |
  | ------- | ----- | ---------------- |
  | A       | float | percent per unit |
  | B       | float |                  |
  | C       | float |                  |
  | D       | float |                  |
  | E       | float |                  |
  | F       | float |                  |
  | G       | float |                  |
  | H       | float |                  |
  | I       | float |                  |
  | J       | float |                  |
  | K       | float |                  |

* `bus_sourceid`
  | columns | dtype  | description                                      |
  | ------- | ------ | ------------------------------------------------ |
  | bus_sid | string | bus source id (in string format ending with " ") |

* `gen_sourceid`
  | columns | dtype  | description                                            |
  | ------- | ------ | ------------------------------------------------------ |
  | gen_sid | string | generator source id (in string format ending with " ") |

* `branch_sourceid`
  | columns    | dtype  | description                                         |
  | ---------- | ------ | --------------------------------------------------- |
  | fbus       | int    | from bus source id                                  |
  | tbus       | int    | target bus source id                                |
  | branch_sid | string | branch source id (in string format ending with " ") |