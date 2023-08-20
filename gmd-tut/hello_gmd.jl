using PowerModelsGMD, Ipopt, JuMP, PowerModels, CSV

include("io.jl")

# net = PowerModels.parse_file("epri21.m")
net = PowerModels.parse_file("case24_ieee_rts_0.m")

# change feature of a node in net

cfg = Dict{String,Any}("output" => Dict{String,Any}("branch_flows" => true))
solver = JuMP.with_optimizer(Ipopt.Optimizer, tol=1e-6, print_level=0)

dc_result = PowerModelsGMD.run_gmd(net, solver)
gmd_bus_df = to_df(net, "gmd_bus", dc_result)
CSV.write("gmd_bus.csv", gmd_bus_df)

result = PowerModelsGMD.run_ac_gmd_mls(net, solver; setting=cfg)
load_df = to_df(net, "load", result)
CSV.write("load.csv", load_df)
