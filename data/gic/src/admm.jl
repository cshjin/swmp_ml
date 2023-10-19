using JuMP, Ipopt

function admm_algorithm(args, m_ac, m_dc, var, pd)

    ## initialization
    nz = length(pd.substations)
    nc = 0
    for line in pd.Line
        if line.config == "gwye-delta" || line.config == "gwye-gwye-auto" || line.config == "gwye-gwye"
            nc += 1
        end
    end

    λ = ones(nz)
    μ = ones(nc)
    zc = zeros(nz)
    zb = zeros(nz)
    Iac = zeros(nc)

    v = zeros(nz + nc)
    u = zeros(nz + nc)
    u_ = zeros(nz + nc)
    w = zeros(nz + nc)


    ρ = args["rho"]
    ϵ = args["eps"]

    if args["resbal"] == 1
        τ = args["tau"]
        β = args["beta"]
    end

    res_io = write_output_admm(args)
    log_title_admm(res_io)

    pd_max = Inf
    iter = 1
    stime_1 = time()
    while pd_max > ϵ && iter <= args["n_iter"]
        stime_2 = time()
        ## store solutions from the previous iteration
        u_ .= u

        ## first-block
        zb = first_block_closed_form(args, zc, λ, ρ)

        activate_z = Vector{Int64}()
        for i = 1:length(zb)
            if zb[i] > 0.95
                push!(activate_z, i)
            end
        end

        ## second-block
        zc, Idc = second_block_gic(pd, m_dc, var, λ, μ, zb, Iac, ρ, nz, nc)

        ## third-block
        Iac = third_block_opf(pd, m_ac, var, μ, Idc, ρ, nc)

        ## dual update
        λ += ρ * (zb - zc)
        μ += ρ * (Idc - Iac)


        ## residuals        
        v = vcat(zb, Idc)
        u = vcat(zc, Iac)
        w = vcat(λ, μ)

        pres = norm(v - u)
        dres = ρ * norm(u - u_)

        pres_normalized = pres / max(norm(u), norm(v))
        dres_normalized = dres / norm(w)
        pd_max = max(pres_normalized, dres_normalized)

        args["iter"] = iter
        args["pres"] = pres_normalized
        args["dres"] = dres_normalized
        args["pdmax"] = pd_max
        args["ρ"] = ρ
        args["periter_time"] = time() - stime_2
        args["elapsed_time"] = time() - stime_1

        log_iteration_admm(res_io, args, activate_z)

        if args["resbal"] == 1
            if pres_normalized > β * dres_normalized
                ρ *= τ
            elseif β * pres_normalized < dres_normalized
                ρ /= τ
            end
            if ρ > args["max_rho"]
                ρ = args["max_rho"]
            end
        end

        iter += 1

    end

    close(res_io)


    return zb
end

function first_block_closed_form(args, zc, λ, ρ)

    # cost 
    c = 0.5 * ρ .+ λ .- ρ .* zc
    # sort
    p = sortperm(c)
    # assign
    B = args["budget"]
    z = zeros(length(zc))
    for idx in p
        if c[idx] < 0 && B > 0.0
            z[idx] = 1
            B -= 1.0
        end
    end
    return z
end

function first_block_closed_form_no_budget(zc, λ, ρ, nz)
    val_0 = 0.5 * ρ * zc .^ 2
    val_1 = λ .+ 0.5 * ρ * (1.0 .- zc) .^ 2

    z = zeros(nz)
    for i in 1:nz
        if val_0[i] > val_1[i]
            z[i] = 1
        end
    end
    return z
end

function second_block_gic(pd, m_dc, var, λ, μ, zb, Iac, ρ, nz, nc)

    obj_curr = JuMP.objective_function(m_dc)
    obj = 0.0
    for i in 1:nz
        obj += -λ[i] * var.z[i] + 0.5 * ρ * (zb[i] - var.z[i])^2
    end
    tmpcnt = 0
    for line in pd.Line
        e = line.line_i
        if line.config == "gwye-delta" || line.config == "gwye-gwye-auto" || line.config == "gwye-gwye"
            tmpcnt += 1
            obj += μ[tmpcnt] * var.Ieff[e] + 0.5 * ρ * (var.Ieff[e] - Iac[tmpcnt])^2
        end
    end

    @objective(m_dc, Min, obj + obj_curr)

    JuMP.optimize!(m_dc)

    z = zeros(nz)
    for i in 1:nz
        z[i] = value(var.z[i])
    end
    I = zeros(nc)
    tmpcnt = 0
    for line in pd.Line
        e = line.line_i
        if line.config == "gwye-delta" || line.config == "gwye-gwye-auto" || line.config == "gwye-gwye"
            tmpcnt += 1
            I[tmpcnt] = value(var.Ieff[e])
        end
    end
    @objective(m_dc, Min, obj_curr)
    return z, I
end

function third_block_opf(pd, m_ac, var, μ, Idc, ρ, nc)

    obj_curr = JuMP.objective_function(m_ac)
    obj = 0.0
    tmpcnt = 0
    for line in pd.Line
        e = line.line_i
        if line.config == "gwye-delta" || line.config == "gwye-gwye-auto" || line.config == "gwye-gwye"
            tmpcnt += 1
            obj += -μ[tmpcnt] * var.Ieff_ac[e] + 0.5 * ρ * (Idc[tmpcnt] - var.Ieff_ac[e])^2
        end
    end

    @objective(m_ac, Min, obj + obj_curr)

    # println("-----------obj=",  JuMP.objective_function(m_ac))
    JuMP.optimize!(m_ac)

    I = zeros(nc)
    tmpcnt = 0
    for line in pd.Line
        e = line.line_i
        if line.config == "gwye-delta" || line.config == "gwye-gwye-auto" || line.config == "gwye-gwye"
            tmpcnt += 1
            I[tmpcnt] = value(var.Ieff_ac[e])
        end
    end

    @objective(m_ac, Min, obj_curr)
    # println("***********obj=", JuMP.objective_function(m_ac))

    return I
end

##### 
function construct_second_block_gic_model(args, var, pd)

    m = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
    set_time_limit_sec(m, args["time_limit"])

    ## variables
    variables_blockers_relax(m, var, pd)
    variables_effective_gic(m, var, pd)
    variables_buses_dc(m, var, pd)
    variables_lines_dc(m, var, pd)


    ## constraints    
    constraints_gic(m, var, pd)
    constraints_effective_gic_complementarity(m, var, pd)
    constraints_gic_balance_with_blockers_var(m, var, pd)


    ## Objective function
    OBJ = 0.0
    @objective(m, Min, OBJ)

    return m
end

function construct_third_block_opf_model(args, var, pd)

    m = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
    set_time_limit_sec(m, args["time_limit"])

    variables_generators(m, var, pd)
    variables_buses(m, var, pd)
    variables_lines(m, var, pd)

    if args["model"] == "ac_rect" || args["model"] == "soc_rect"
        variables_rect(m, var, pd)
    end

    constraints_power_balance(m, var, pd)
    constraints_power_flow(m, var, pd)
    constraints_angle_difference(m, var, pd)
    constraints_thermal_limit(m, var, pd)

    ## Effective GIC and dqloss
    var.Ieff_ac = Dict()
    for line in pd.Line
        e = line.line_i
        if line.config == "gwye-delta" || line.config == "gwye-gwye-auto" || line.config == "gwye-gwye"
            var.Ieff_ac[e] = @variable(m, lower_bound = 0.0, upper_bound = pd.Immax, base_name = "Ieff_ac[$(e)]")
        end
    end
    for bus in pd.Bus
        i = bus.bus_i
        LE = 0
        for line in pd.Line
            e = line.line_i
            if line.config == "gwye-delta" || line.config == "gwye-gwye-auto" || line.config == "gwye-gwye"
                if i == line.fbus || i == line.tbus
                    baseKV = pd.Bus[pd.Bus_index2id[line.hi_bus]].baseKV
                    ibase = pd.baseMVA * 1000.0 * sqrt(2.0) / (baseKV * sqrt(3.0))
                    K = line.gmd_k * pd.baseMVA / ibase
                    LE = LE + K * var.v[i] * var.Ieff_ac[e] / (3.0 * pd.baseMVA)
                end
            end
        end
        @NLconstraint(m, var.dqloss[i] == LE)
    end


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

    return m
end

function write_output_admm(args)

    dir = args["output_dir_name"]

    if isdir(dir) == false
        mkpath(dir)
    end

    path = dir * args["output_file_name"] * "_log"
    outfile = get_new_filename(path, ".txt")
    res_io = open(outfile * ".txt", "w")
    return res_io

end

function log_title_admm(res_io)
    @printf(res_io,
        "%10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s \n",
        "iter",
        "pres",
        "dres",
        "pdmax",
        "eps",
        "ρ",
        "per[s]",
        "elpsd[s]"
    )

    flush(res_io)
    @printf(
        "%10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s \n",
        "iter",
        "pres",
        "dres",
        "pdmax",
        "eps",
        "ρ",
        "per[s]",
        "elpsd[s]"
    )

end

function log_iteration_admm(res_io, args, activate_z)

    @printf(res_io,
        "%10d  %10.4e  %10.4e  %10.4e  %10.4e  %10.4e  %10.4e  %10.4e  %10d  %10s \n",
        args["iter"],
        args["pres"],
        args["dres"],
        args["pdmax"],
        args["eps"],
        args["ρ"],
        args["periter_time"],
        args["elapsed_time"],
        length(activate_z),
        activate_z
    )
    flush(res_io)
    @printf(
        "%10d  %10.4e  %10.4e  %10.4e  %10.4e  %10.4e  %10.4e  %10.4e  %10d  %10s \n",
        args["iter"],
        args["pres"],
        args["dres"],
        args["pdmax"],
        args["eps"],
        args["ρ"],
        args["periter_time"],
        args["elapsed_time"],
        length(activate_z),
        "-"
    )
end