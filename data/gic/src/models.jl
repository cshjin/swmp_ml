include("variables.jl")
include("objective.jl")
include("constraints.jl")

function get_invariant_model(args, m, var, pd)

    variables_generators(m, var, pd)
    variables_buses(m, var, pd)
    variables_lines(m, var, pd)
    variables_effective_gic(m, var, pd)
    variables_buses_dc(m, var, pd)
    variables_lines_dc(m, var, pd)

    if args["model"] == "ac_rect" || args["model"] == "soc_rect"
        variables_rect(m, var, pd)
    end

    constraints_power_balance(m, var, pd)
    constraints_power_flow(m, var, pd)
    constraints_angle_difference(m, var, pd)
    constraints_thermal_limit(m, var, pd)
    constraints_dqloss(m, var, pd)
    constraints_gic(m, var, pd)
    constraints_effective_gic_complementarity(m, var, pd)

    if args["model"] == "ac_polar"
        constraints_ac_polar(m, var, pd)
    end
    if args["model"] == "soc_polar"
        constraints_soc_polar(m, var, pd)
    end
    if args["model"] == "ac_rect"
        constraints_ac_rect(m, var, pd)
    end
    if args["model"] == "soc_rect"
        constraints_soc_rect(m, var, pd)
    end

    return m
end

function add_blockers_objective_model_minlp(args, m, var, pd)
    variables_blockers_binary(m, var, pd)
    constraints_gic_balance_with_blockers_var(m, var, pd)
    constraints_blockers(args, m, var, pd)
    construct_objective(m, var, pd)
    return m
end


function add_blockers_model_nlp(m, var, pd)

    m, c1 = constraints_gic_balance_with_blockers_fix(m, var, pd)
    construct_objective(m, var, pd)

    return m, c1
end

function remove_blockers_model_nlp(m, c1)

    for (i, _) in c1
        delete(m, c1[i])
    end

    return m
end

function add_blockers_model_trust(m, var, pd)
    variables_blockers_relax(m, var, pd)
    m, c1 = constraints_gic_balance_with_blockers_trust(m, var, pd)
    construct_objective(m, var, pd)

    return m, c1
end

function remove_blockers_model_trust(m, c1, pd)

    for (i, _) in c1
        delete(m, c1[i])
    end

    for (i, _) in pd.trust_consensus
        delete(m, pd.trust_consensus[i])
    end

    return m
end

