function construct_objective(m, var, pd)
    ## Objective function
    OBJ = 0
    for gen in pd.Gen
        k = gen.gen_i
        OBJ = OBJ + gen.cF1 * var.fp[k] + gen.cF2 * var.fp[k]^2
    end

    for bus in pd.Bus
        i = bus.bus_i
        OBJ = OBJ + pd.penalty1 * (var.lpp[i] + var.lpm[i] + var.lqp[i] + var.lqm[i])
    end

    @objective(m, Min, OBJ)
end

