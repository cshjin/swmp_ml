using ArgParse, Printf
using JuMP, Ipopt

include("./src/gic.jl")
using .gic
import Random

function parse_commandline()
  s = ArgParseSettings()
  @add_arg_table s begin
    "--network"
    arg_type = String
    default = "ACTIVSg2000"  ## epri21, uiuc150        
    "--model"
    arg_type = String
    default = "ac_rect"  ## ac_polar, ac_rect, soc_polar, soc_rect        
    "--optimizer"
    arg_type = String
    default = "admm"
    "--rho"    # step size
    arg_type = Float64
    default = 1e2
    "--eps"    # termination
    arg_type = Float64
    default = 5e-3
    "--resbal"    # residual balancing
    arg_type = Int64
    default = 1
    "--tau"    # residual balancing
    arg_type = Float64
    default = 2.0
    "--beta"    # residual balancing
    arg_type = Float64
    default = 5.0
    "--max_rho"    # residual balancing
    arg_type = Float64
    default = 1e3
    "--n_iter"
    arg_type = Int64
    default = 5000
    "--time_limit"
    arg_type = Float64
    default = 3600.0
    "--efield_mag"
    arg_type = Float64
    default = 20.0  ##  [5.0, 10.0, 15.0, 20.0]
    "--efield_dir"
    arg_type = Float64
    default = 45.0 ##  [45.0, 90.0, 135.0]
    "--output_dir_name"
    arg_type = String
    default = "./tmp/"
    "--run_id"
    arg_type = Int64
    default = 1
  end
  return parse_args(s)
end

args = parse_commandline()

## budget for blockers 
if args["network"] == "epri21"
  args["budget"] = 3.0
elseif args["network"] == "uiuc150"
  args["budget"] = 30.0
elseif args["network"] == "ACTIVSg2000"
  args["budget"] = 100.0
else
  error("unknown network")
end

## filename
# args["output_dir_name"] = "./outputs/$(args["network"])_$(args["model"])/$(args["optimizer"])/mag_$(args["efield_mag"])_dir_$(args["efield_dir"])_budget_$(args["budget"])_rho_$(args["rho"])_eps_$(args["eps"])_resbal_$(args["resbal"])/"

# args["output_file_name"] = "$(args["network"])_$(args["model"])_$(args["optimizer"])_$(args["rho"])_$(args["n_iter"])_$(args["efield_mag"])_$(args["efield_dir"])"

args["output_dir_name"] = "$(args["output_dir_name"])$(args["network"])/"

args["output_file_name"] = @sprintf("%03d_%s_%s_%s_%s_%.1f_%.1f",
  args["run_id"],
  args["network"],
  args["model"],
  args["optimizer"],
  # args["lr"],
  # args["n_samples"],
  args["n_iter"],
  args["efield_mag"],
  args["efield_dir"])


## data
pd = gic.PowerData()
gic.read_data(args, pd)

## ! NOTE: add random perturbations to bus load (pd/qd) 
for bus in pd.Bus
  i = bus.bus_i
  if bus.pd !== 0 && bus.qd !== 0
    bus.pd *= rand(0.8:0.05:1.2)
    bus.qd *= rand(0.8:0.05:1.2)
  end
end

## optimizer and models
var = gic.Variables()

m_dc = gic.construct_second_block_gic_model(args, var, pd)
m_ac = gic.construct_third_block_opf_model(args, var, pd)

## admm
stime = time()
z = gic.admm_algorithm(args, m_ac, m_dc, var, pd)
args["elapsed_time"] = time() - stime

## Evaluation of "z" using AC_POLAR
m = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
set_time_limit_sec(m, args["time_limit"])
var = gic.Variables()
m = gic.get_invariant_model(args, m, var, pd)
## set "z"
for idx in 1:length(z)
  gmd_bus_idx = pd.substations[idx]
  pd.z[gmd_bus_idx] = z[idx]
end
m, _ = gic.add_blockers_model_nlp(m, var, pd)

JuMP.optimize!(m)



## log
gic.write_summary(args, m, var, pd)
# gic.write_gen(args, var, pd)
# gic.write_bus(args, var, pd)
# gic.write_line(args, var, pd)
# gic.write_bus_dc(args, var, pd)
# gic.write_line_dc(args, var, pd)
gic.write_json(args, var, pd)


