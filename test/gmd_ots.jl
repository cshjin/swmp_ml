@testset "Test AC GMD OTS" begin

    # -- EPRI21 -- #
    # EPRI21 - 19-bus case

    @testset "EPRI21 case" begin

        # TODO: fix EPRI21 LOCALLY_INFEASIBLE issue
        # TODO: add tests

        # casename = "../test/data/epri21_ots.m"
        # case = PowerModels.parse_file(casename)
        # result = PowerModelsGMD.run_ac_gmd_ots(casename, juniper_optimizer)

        # @test result["termination_status"] == PowerModels.LOCALLY_SOLVED || result["termination_status"] == PowerModels.OPTIMAL
        # @test isapprox(result["objective"], 2.46069149389163e6; atol = 1e-1, rtol = 1e-3)

    end


    # -- OTS-TEST -- #
    # OTS-TEST - 57-bus case

    @testset "OTS-Test case" begin

        # TODO: fix OTS-TEST LOCALLY_INFEASIBLE issue
        # TODO: add tests

        # casename = "../test/data/ots_test.m"
        # case = PowerModels.parse_file(casename)
        # result = PowerModelsGMD.run_ac_gmd_ots(casename, juniper_optimizer)

        # @test result["termination_status"] == PowerModels.LOCALLY_SOLVED || result["termination_status"] == PowerModels.OPTIMAL
        # @test isapprox(result["objective"], 2.46069149389163e6; atol = 1e-1, rtol = 1e-3)

    end

end


