using JuMP, SCIP
using LinearAlgebra

function trust_region_algorithm(args, m, var, pd)

    ## Initialization           
    z = zeros(length(pd.substations))

    res_io = write_output_trust(args)
    log_title_trust(res_io)

    delta = args["radius_init"]
    rho_ = args["radius_threshold"]

    ## evaluation
    obj, grad = compute_obj_grad(m, var, pd, z)

    ##      
    n = length(z)
    sub = JuMP.direct_model(SCIP.Optimizer())
    JuMP.set_silent(sub)
    @variable(sub, x[1:n], Bin)
    @constraint(sub, sum(x[i] for i = 1:n) <= args["budget"])

    iteration = 0
    stime_1 = time()
    while delta >= 1.0
        iteration += 1
        stime_2 = time()
        z_hat = solve_trust_region_subproblem(sub, x, n, grad, z, delta)
        args["time_ip"] = time() - stime_2

        """ termination """
        if LinearAlgebra.norm(z_hat - z, 1) <= 1e-8
            println("------termination: z_hat = z ----")
            break
        end
        if iteration > args["n_iter"]
            print("------termination: total_iteration ----")
            break
        end

        ## evaluation
        stime_3 = time()
        obj_hat, grad_hat = compute_obj_grad(m, var, pd, z_hat)
        args["time_eval"] = time() - stime_3

        ## update        
        stime_4 = time()
        rho = (obj - obj_hat) / LinearAlgebra.dot(grad, z - z_hat)
        if rho > 0.0
            ## precalculation
            norm_temp = LinearAlgebra.norm(z_hat - z, 1)

            ## update
            z = deepcopy(z_hat)
            obj = deepcopy(obj_hat)
            grad = deepcopy(grad_hat)

            if rho > rho_ && abs(norm_temp - delta) <= 1e-8
                delta = delta * 2.0
            end
        else
            delta = floor(delta / 2.0)
        end
        args["time_up"] = time() - stime_4
        args["elapsed_time"] = time() - stime_1
        # store
        args["iter"] = iteration
        args["delta"] = delta
        args["rho"] = rho
        args["obj"] = obj
        log_iteration_trust(res_io, args)
    end

    return z
end

function solve_trust_region_subproblem(sub, x, n, grad, z, delta)

    ## objective
    @objective(sub, Min, sum(grad[i] * x[i] for i = 1:n))
    ## constraints
    c = @constraint(sub, sum((1.0 - 2.0 * z[i]) * x[i] for i = 1:n) <= delta - sum(z[i] for i = 1:n))

    JuMP.optimize!(sub)

    z_hat = []
    for i = 1:n
        push!(z_hat, JuMP.value(x[i]))
    end

    ## delete constraint "c"
    JuMP.delete(sub, c)

    return z_hat
end

function compute_obj_grad(m, var, pd, z)

    ## set "z"
    for idx in 1:length(z)
        gmd_bus_idx = pd.substations[idx]
        pd.z[gmd_bus_idx] = z[idx]
    end

    m, c1 = gic.add_blockers_model_trust(m, var, pd)


    ## solve
    JuMP.optimize!(m)

    obj = JuMP.objective_value(m)

    grad = []
    for gmdbus in pd.GMDBus
        i = gmdbus.GMDbus_i
        if gmdbus.g_gnd > 0.0 ## positive for substations
            push!(grad, JuMP.dual(pd.trust_consensus[i]))
        end
    end

    ##
    m = remove_blockers_model_trust(m, c1, pd)

    return obj, grad
end

function write_output_trust(args)

    dir = args["output_dir_name"]

    if isdir(dir) == false
        mkpath(dir)
    end

    path = dir * args["output_file_name"] * "_log"
    outfile = get_new_filename(path, ".txt")
    res_io = open(outfile * ".txt", "w")
    return res_io

end
function log_title_trust(res_io)
    @printf(res_io,
        "%10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s   \n",
        "iter",
        "objval",
        "t_radius",
        "rho",
        "ip[s] ",
        "eval[s]",
        "update[s]",
        "elpsd[s]"
    )

    flush(res_io)
    @printf(
        "%10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s   \n",
        "iter",
        "objval",
        "t_radius",
        "rho",
        "ip[s] ",
        "eval[s]",
        "update[s]",
        "elpsd[s]"
    )

end
function log_iteration_trust(res_io, args)

    @printf(res_io,
        "%10d  %10.4e  %10.4e  %10.4e  %10.4e  %10.4e  %10.4e  %10.4e     \n",
        args["iter"],
        args["obj"],
        args["delta"],
        args["rho"],
        args["time_ip"],
        args["time_eval"],
        args["time_up"],
        args["elapsed_time"],
    )
    flush(res_io)
    @printf(
        "%10d  %10.4e  %10.4e  %10.4e  %10.4e  %10.4e  %10.4e  %10.4e     \n",
        args["iter"],
        args["obj"],
        args["delta"],
        args["rho"],
        args["time_ip"],
        args["time_eval"],
        args["time_up"],
        args["elapsed_time"],
    )
end