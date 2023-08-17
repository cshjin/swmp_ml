module gic
    using XLSX ## read data
    using JuMP  ## construct mathematical models
    using Ipopt ## solve nonlinear programs

    using DataFrames ## write solution
    using Printf
    using JSON

    include("read.jl")
    include("log.jl")
    include("structure.jl")
    include("models.jl")
    include("heuristics.jl")

end
