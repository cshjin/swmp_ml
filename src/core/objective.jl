# --- Objectives --- #


"OBJECTIVE: computes a load shed cost"
function calc_load_shed_cost(pm::_PM.AbstractPowerModel)

    max_cost = 0
    for (n, nw_ref) in _PM.nws(pm)
        for (i,gen) in nw_ref[:gen]
            if gen["pmax"] != 0
                cost_mw = (get(gen["cost"], 1, 0.0)*gen["pmax"]^2 + get(gen["cost"], 2, 0.0)*gen["pmax"]) / gen["pmax"] + get(gen["cost"], 3, 0.0)
                max_cost = max(max_cost, cost_mw)
            end
        end
    end

    return max_cost * 2.0

end


"OBJECTIVE: OPF objective"
function objective_gmd_min_fuel(pm::_PM.AbstractPowerModel)

    #@assert all(!_PM.ismulticonductor(pm) for n in _PM.nws(pm))

    #i_dc_mag = Dict(n => pm.var[:nw][n][:i_dc_mag] for n in nws) #pm.var[:i_dc_mag]
    #pg = Dict(n => pm.var[:nw][n][:pg] for n in nws) #pm.var[:pg]

    return JuMP.@objective(pm.model, Min,
    sum(
        sum(
            get(gen["cost"], 1, 0.0)*sum( _PM.var(pm, n, c, :pg, i) for c in _PM.conductor_ids(pm, n))^2 +
            get(gen["cost"], 2, 0.0)*sum( _PM.var(pm, n, c, :pg, i) for c in _PM.conductor_ids(pm, n)) +
            get(gen["cost"], 3, 0.0)
        for (i,gen) in nw_ref[:gen]) +
        sum(
            sum( _PM.var(pm, n, c, :i_dc_mag, i)^2 for c in _PM.conductor_ids(pm, n) )
        for (i,branch) in nw_ref[:branch])
    for (n, nw_ref) in _PM.nws(pm))
    )

end


"OBJECTIVE: SSE -- keep generators as close as possible to original setpoint"
function objective_gmd_min_error(pm::_PM.AbstractPowerModel)

    @assert all(!_PM.ismulticonductor(pm) for n in _PM.nws(pm))

    #i_dc_mag = Dict(n => pm.var[:nw][n][:i_dc_mag] for n in nws) #pm.var[:i_dc_mag]
    #pg = Dict(n => pm.var[:nw][n][:pg] for n in nws) # pm.var[:pg]
    #qg = Dict(n => pm.var[:nw][n][:qg] for n in nws) # pm.var[:qg]
    #z_demand = Dict(n => pm.var[:nw][n][:z_demand] for n in nws) # pm.var[:z_demand]

    M_p = Dict(n => max([gen["pmax"] for (i,gen) in nw_ref[:gen]]) for (n, nw_ref) in _PM.nws(pm))

    # return JuMP.@objective(pm.model, Min, sum{ i_dc_mag[i]^2, i in keys(pm.ref[:branch])})
    # return JuMP.@objective(pm.model, Min, sum(get(gen["cost"], 1, 0.0)*pg[i]^2 + get(gen["cost"], 2, 0.0)*pg[i] + get(gen["cost"], 3, 0.0) for (i,gen) in pm.ref[:gen]) )
    return JuMP.@objective(pm.model, Min,
    sum(
        sum(
            (gen["pg"] - _PM.var(pm, :pg, i, nw=n))^2 + (gen["qg"] - _PM.var(pm, :qg, i, nw=n))^2
        for (i,gen) in _PM.ref(pm, n, :gen)) +
        sum(
            _PM.var(pm, :i_dc_mag, i, nw=n)^2
        for (i,branch) in _PM.ref(pm, n, :branch)) +
        sum(
            -100.0*M_p^2*_PM.var(pm, :z_demand, i, nw=n)
        for (i,load) in _PM.ref(pm, n, :load))
    for (n, nw_ref) in _PM.nws(pm))
    )

end


"OBJECTIVE: minimizes load shedding and fuel cost"
function objective_gmd_min_ls(pm::_PM.AbstractPowerModel)

    @assert all(!_PM.ismulticonductor(pm) for n in _PM.nws(pm))

    #pg = Dict(n => pm.var[:nw][n][:pg] for n in nws)
    #pd = Dict(n => pm.var[:nw][n][:pd] for n in nws)
    #qd = Dict(n => pm.var[:nw][n][:qd] for n in nws)

    shed_cost = calc_load_shed_cost(pm)

    return JuMP.@objective(pm.model, Min,
    sum(
        sum(
            get(gen["cost"], 1, 0.0)*_PM.var(pm, :pg, i, nw=n)^2 +
            get(gen["cost"], 2, 0.0)*_PM.var(pm, :pg, i, nw=n) +
            get(gen["cost"], 3, 0.0)
        for (i,gen) in nw_ref[:gen]) +
        sum(
            shed_cost*(_PM.var(pm, :pd, i, nw=n) + _PM.var(pm, :qd, i, nw=n))
        for (i,load) in nw_ref[:load])
    for (n, nw_ref) in _PM.nws(pm))
    )

end


"OBJECTIVE: minimizes load shedding and fuel cost"
function objective_gmd_min_ls_on_off(pm::_PM.AbstractPowerModel)

    @assert all(!_PM.ismulticonductor(pm) for n in _PM.nws(pm))

    #pg     = Dict(n => pm.var[:nw][n][:pg] for n in nws)
    #pd     = Dict(n => pm.var[:nw][n][:pd] for n in nws)
    #qd     = Dict(n => pm.var[:nw][n][:qd] for n in nws)
    #z      = Dict(n => pm.var[:nw][n][:gen_z] for n in nws)
    #pg_sqr = Dict(n => pm.var[:nw][n][:pg_sqr] for n in nws)

    shed_cost = calc_load_shed_cost(pm)

    return JuMP.@objective(pm.model, Min,
    sum(
        sum(
            get(gen["cost"], 1, 0.0)*_PM.var(pm, :pg_sqr, i, nw=n) +
            get(gen["cost"], 2, 0.0)*_PM.var(pm, :pg, i, nw=n) +
            get(gen["cost"], 3, 0.0)*_PM.var(pm, :gen_z, i, nw=n)
        for (i,gen) in nw_ref[:gen]) +
        sum(
            shed_cost*(_PM.var(pm, :pd, i, nw=n) + _PM.var(pm, :qd, i, nw=n))
        for (i,load) in nw_ref[:load])
    for (n, nw_ref) in _PM.nws(pm))
    )

end


"OBJECTIVE: minimizes transfomer heating caused by GMD"
function objective_gmd_min_transformer_heating(pm::_PM.AbstractPowerModel)

    #@assert all(!_PM.ismulticonductor(pm) for n in _PM.nws(pm))

    #i_dc_mag = Dict(n => pm.var[:nw][n][:i_dc_mag] for n in nws) #pm.var[:i_dc_mag]
    #pg = Dict(n => pm.var[:nw][n][:pg] for n in nws) #pm.var[:pg]

    # TODO: add i_dc_mag minimization

    return JuMP.@objective(pm.model, Min,
    sum(
        sum(
            sum( _PM.var(pm, n, c, :hsa, i) for c in _PM.conductor_ids(pm, n) )
        # for (i,branch) in _PM.nw_ref[:branch])
        for (i,branch) in nw_ref[:branch])
    for (n, nw_ref) in _PM.nws(pm))
    )

end


