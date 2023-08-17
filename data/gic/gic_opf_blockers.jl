using ArgParse, Printf
using JuMP, Ipopt, Juniper
using MathOptInterface, SCIP

include("./src/gic.jl")
using .gic

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin        
    "--network"            
        arg_type = String
        default = "epri21"  ## epri21, uiuc150        
    "--model"            
        arg_type = String
        default = "ac_polar"  ## ac_polar, ac_rect, soc_polar, soc_rect        
    "--optimizer"            
        arg_type = String
        default = "juniper"  ## juniper, scip 
    "--time_limit"            
        arg_type = Float64
        default = 3600.0 
    "--efield_mag"            
        arg_type = Float64
        default = 10.0  ##  [5.0, 10.0, 15.0, 20.0]    
    "--efield_dir"            
        arg_type = Float64
        default = 45.0 ##  [45.0, 90.0, 135.0]
    "--tot_num_blockers"            
        arg_type = Int64
        default = 10 ##   
    "--output_dir_name"            
        arg_type = String
        default = "./outputs/"
    "--run_id"
        arg_type = Int64
        default = 1
    end
    return parse_args(s)
end            

args = parse_commandline()

## data
pd = gic.PowerData()
gic.read_data(args, pd)

## changing load (if necessary)
# for bus in pd.Bus
#     i = bus.bus_i
#     # println(i, "  pd=", bus.pd, "  qd=", bus.qd )
#     bus.pd *= 1.0
#     bus.qd *= 1.0
#     # println(i, "  pd=", bus.pd, "  qd=", bus.qd )
# end 

## NOTE: add random perturbations to bus load (pd/qd) 
for bus in pd.Bus
    i = bus.bus_i
    if bus.pd !== 0 && bus.qd !== 0
        bus.pd *= rand(0.8:0.05:1.2)
        bus.qd *= rand(0.8:0.05:1.2)
    end
end

## optimizer and model
if args["optimizer"] == "juniper"
    optimizer = Juniper.Optimizer
    nl_solver = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
    m = Model(optimizer_with_attributes(optimizer, "nl_solver"=>nl_solver))   
elseif args["optimizer"] == "scip"
    optimizer = SCIP.Optimizer 
    m = Model(optimizer_with_attributes(optimizer))
end

set_time_limit_sec(m, args["time_limit"])

var = gic.Variables()  
if args["model"] == "ac_polar"
    m = gic.construct_gic_blockers_ac_polar_model(m, var, pd)
end
if args["model"] == "ac_rect"
    m = gic.construct_gic_blockers_ac_rect_model(m, var, pd)
end
if args["model"] == "soc_polar"
    m = gic.construct_gic_blockers_soc_polar_model(m, var, pd)
end
if args["model"] == "soc_rect"
    m = gic.construct_gic_blockers_soc_rect_model(m, var, pd)
end

stime = time()
JuMP.optimize!(m)
args["elapsed_time"] = time() - stime
println("Termination status=", termination_status(m))

## log
# args["output_file_name"]="$(args["run_id"])_$(args["optimizer"])_$(args["network"])_$(args["model"])_$(args["tot_num_blockers"])_$(args["efield_mag"])_$(args["efield_dir"])"
args["output_file_name"] = @sprintf("%03d_%s_%s_%s_%02d_%.1f_%.1f", args["run_id"],
                                                                    args["optimizer"],
                                                                    args["network"],
                                                                    args["model"],
                                                                    args["tot_num_blockers"],
                                                                    args["efield_mag"],
                                                                    args["efield_dir"])
# gic.write_summary(args, m, var, pd) 
# gic.write_gen(args, var, pd) 
# gic.write_bus(args, var, pd) 
# gic.write_line(args, var, pd) 
# gic.write_bus_dc(args, var, pd) 
# gic.write_line_dc(args, var, pd) 


gic.write_json(args, var, pd)