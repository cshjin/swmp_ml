using ArgParse, Printf
using JuMP, Ipopt

include("./src/gic.jl")
using .gic
import Random

# Random.seed!(1)

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--network"
        arg_type = String
        default = "epri21"  ## epri21, uiuc150
        "--model"
        arg_type = String
        default = "ac_rect"  ## ac_polar, ac_rect, soc_polar, soc_rect
        "--optimizer"
        arg_type = String
        default = "stochastic"
        "--n_samples"
        arg_type = Int64
        default = 10
        "--n_iter"
        arg_type = Int64
        default = 1
        "--lr"
        arg_type = Float64
        default = 1e-6
        "--time_limit"
        arg_type = Float64
        default = 3600.0
        "--efield_mag"
        arg_type = Float64
        default = 5.0  ##  [5.0, 10.0, 15.0, 20.0]
        "--efield_dir"
        arg_type = Float64
        default = 45.0 ##  [45.0, 90.0, 135.0]
        "--output_dir_name"
        arg_type = String
        default = "./results/"
        "--run_id"
        arg_type = Int64
        default = 1
    end
    return parse_args(s)
end

args = parse_commandline()

## budget for blockers 
if args["network"] == "epri21"
    args["budget"] = 2.0
elseif args["network"] == "uiuc150"
    args["budget"] = 25.0
end

## filename
# args["output_dir_name"] = "./outputs/$(args["network"])_$(args["model"])/$(args["optimizer"])_budget_$(args["budget"])_lr_$(args["lr"])/mag_$(args["efield_mag"])_dir_$(args["efield_dir"])_budget_$(args["budget"])/"
# args["output_file_name"] = "$(args["network"])_$(args["model"])_$(args["optimizer"])_$(args["lr"])_$(args["n_samples"])_$(args["n_iter"])_$(args["efield_mag"])_$(args["efield_dir"])"

args["output_dir_name"] = args["output_dir_name"]

args["output_file_name"] = @sprintf("%03d_%s_%s_%s_%s_%s_%s_%.1f_%.1f", args["run_id"],
    args["network"],
    args["model"],
    args["optimizer"],
    args["lr"],
    args["n_samples"],
    args["n_iter"],
    args["efield_mag"],
    args["efield_dir"])

## data
pd = gic.PowerData()
gic.read_data(args, pd)

## NOTE: add random perturbations to bus load (pd/qd) 
for bus in pd.Bus
    i = bus.bus_i
    if bus.pd !== 0 && bus.qd !== 0
        bus.pd *= rand(0.8:0.05:1.2)
        bus.qd *= rand(0.8:0.05:1.2)
    end
end

## optimizer and model
m = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
set_time_limit_sec(m, args["time_limit"])

var = gic.Variables()
m = gic.get_invariant_model(args, m, var, pd)

## stochastic learning approach
θ = gic.projected_sgd(args, m, var, pd)
# println("θ=", θ)

## find the best scenario 
if args["cnt_01"] > 1e-6
    z = gic.find_best_scenario(args, m, var, pd, θ)
else
    z = θ
end

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
# gic.write_summary(args, m, var, pd)
# gic.write_gen(args, var, pd)
# gic.write_bus(args, var, pd)
# gic.write_line(args, var, pd)
gic.write_bus_dc(args, var, pd)
# gic.write_line_dc(args, var, pd)


