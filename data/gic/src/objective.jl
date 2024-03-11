function construct_objective(m, var, pd)
    ## Objective function
    OBJ = 0.0
    for gen in pd.Gen
        k = gen.gen_i
        OBJ += gen.cF1 * var.fp[k] + gen.cF2 * var.fp[k]^2
    end
    for bus in pd.Bus
        i = bus.bus_i
        OBJ += pd.penalty1 * (var.lpp[i] + var.lpm[i] + var.lqp[i] + var.lqm[i])
    end
    @objective(m, Min, OBJ)
end


function construct_objective_var(m, var, pd)
    ## Objective function
    OBJ = 0.0
    for gen in pd.Gen
        k = gen.gen_i
        OBJ += gen.cF1 * var.fp[k] + gen.cF2 * var.fp[k]^2
    end
    for bus in pd.Bus
        i = bus.bus_i
        OBJ += pd.penalty1 * (var.lpp[i] + var.lpm[i] + var.lqp[i] + var.lqm[i])
    end

    for gmdbus in pd.GMDBus
        i = gmdbus.GMDbus_i
        if gmdbus.g_gnd > 0.0 ## positive for substations
            OBJ += pd.penalty2 * var.z[i]
        end
    end
    @objective(m, Min, OBJ)
end


function construct_objective_fix(m, var, pd)
    ## Objective function
    OBJ = 0.0
    for gen in pd.Gen
        k = gen.gen_i
        OBJ += gen.cF1 * var.fp[k] + gen.cF2 * var.fp[k]^2
    end
    for bus in pd.Bus
        i = bus.bus_i
        OBJ += pd.penalty1 * (var.lpp[i] + var.lpm[i] + var.lqp[i] + var.lqm[i])
    end
    for gmdbus in pd.GMDBus
        i = gmdbus.GMDbus_i
        if gmdbus.g_gnd > 0.0 ## positive for substations
            OBJ += pd.penalty2 * pd.z[i]
        end
    end

    @objective(m, Min, OBJ)
end

