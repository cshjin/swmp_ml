
using PowerModelsGMD
const _PMGMD = PowerModelsGMD

import InfrastructureModels
const _IM = InfrastructureModels

import PowerModels
const _PM = PowerModels

import JSON

import JuMP

import Ipopt

import Juniper

import LinearAlgebra

import SparseArrays

using Test

import Memento
Memento.setlevel!(Memento.getlogger(_PMGMD), "error")
Memento.setlevel!(Memento.getlogger(_IM), "error")
Memento.setlevel!(Memento.getlogger(_PM), "error")

_PMGMD.logger_config!("error")
const TESTLOG = Memento.getlogger(_PMGMD)
Memento.setlevel!(TESTLOG, "error")

ipopt_solver = JuMP.optimizer_with_attributes(Ipopt.Optimizer, "tol" => 1e-4, "print_level" => 0, "sb" => "yes")
juniper_solver = JuMP.optimizer_with_attributes(Juniper.Optimizer, "nl_solver" => _PM.optimizer_with_attributes(Ipopt.Optimizer, "tol" => 1e-4, "print_level" => 0, "sb" => "yes"), "log_levels" => [])
setting = Dict{String,Any}("output" => Dict{String,Any}("branch_flows" => true))



data = "../test/data/matpower/b4gic.m"

case = _PM.parse_file(data)

sol = _PMGMD.solve_gmd(case)

max_error = 1e-2

@testset "linear solve of gmd" begin
	@testset "auto transformers" begin
	end
	@testset "y-d transformers" begin
		 @test isapprox(sol["solution"]["qloss"]["2"], 37.220674406003006, rtol=max_error) || isapprox(sol["solution"]["qloss"]["2"], 37.313418981734294, rtol=max_error) 
		 @test isapprox(sol["solution"]["ieff"]["2"], 22.09875679, rtol=max_error)
		 @test isapprox(sol["solution"]["qloss"]["3"], 37.15240659861487, rtol=max_error) || isapprox(sol["solution"]["qloss"]["3"], 37.2689694, rtol=max_error) 
		 @test isapprox(sol["solution"]["ieff"]["3"], 22.09875679, rtol=max_error)
	end
	@testset "y-y transformers" begin
	end
	@testset "d-d transformers" begin
	end
	@testset "lines" begin
		 @test isapprox(sol["solution"]["qloss"]["1"], 0.0, rtol=max_error)
		 @test isapprox(sol["solution"]["ieff"]["1"], 0.0, rtol=max_error)
	end
end