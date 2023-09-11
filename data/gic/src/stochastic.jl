using Distributions
using LinearAlgebra
using Combinatorics

function projected_sgd(args, m, var, pd)

    ## Initialization
    θ = zeros(length(pd.substations)) .+ 0.5

    N = args["n_samples"]
    z_list = []

    # res_io = write_output_stochastic(args)
    # log_title_stochastic(res_io)

    stime_1 = time()
    for t = 1:args["n_iter"]
        stime_2 = time()

        η = args["lr"] / t

        ## sample "z" such that it satisfies the bueget constraint
        z_scenario, z_list = sample_z_scenario(args, N, θ, pd, z_list)


        obj_list = []
        term_list = []
        grad = 0.0

        for s = 1:length(z_scenario)

            z = z_scenario[s]

            ## 1)
            obj, term = compute_objective_value(m, var, pd, z)

            push!(obj_list, obj)
            push!(term_list, term)

            ## 2) 
            p_grad = grad_log_pmf(z, θ)

            grad = grad .+ obj * p_grad
        end

        ## termination
        if length(z_scenario) > 0.0
            grad = grad ./ length(z_scenario)

            ## update
            θ = θ - η .* grad
            θ = gic.projection_by_truncation(θ)

            grad_project = gic.projection_by_truncation(grad)
        end

        ## store
        cnt_0 = 0
        cnt_1 = 0
        cnt_01 = 0
        for theta_element in θ
            if theta_element < 1e-10
                cnt_0 += 1
            elseif theta_element > 0.99999
                cnt_1 += 1
            else
                cnt_01 += 1
            end
        end
        args["iter"] = t
        args["cnt_0"] = cnt_0
        args["cnt_1"] = cnt_1
        args["cnt_01"] = cnt_01
        args["η"] = η
        args["θ"] = round.(θ, digits=6)
        args["grad_norm"] = LinearAlgebra.norm(grad_project)
        args["periter_time"] = time() - stime_2
        args["elapsed_time"] = time() - stime_1
        args["nsamples_exp"] = length(z_list)

        # log_iteration_stochastic(res_io, args)


        ## termination
        if args["grad_norm"] <= 1e-6 || args["nsamples_exp"] >= 2^(length(pd.substations))
            break
        end

    end

    # close(res_io)

    return θ
end


function sample_z_scenario(args, N, θ, pd, z_list)


    ## count the number of "θ[idx] = 1.0"    
    θ1_indices = []
    for idx in 1:length(θ)
        if θ[idx] > 0.9999
            push!(θ1_indices, idx)
        end
    end

    z_scenario = []
    ## case 1: if the number of "θ[idx] = 1.0" >= args["budget"], the number of samples is set to be tmpcnt choose B    
    if length(θ1_indices) >= args["budget"]
        for list in collect(combinations(θ1_indices, Int(args["budget"])))
            z = zeros(length(pd.substations))
            for idx in list
                z[idx] = 1.0
            end
            push!(z_scenario, z)

        end
    else
        ## p is the sorted indices and θ[p] is the sorted values
        p = sortperm(θ, rev=true)

        for _ = 1:N

            iter = 0
            while true
                iter += 1
                z = zeros(length(pd.substations))
                B = args["budget"]
                for idx in p
                    z[idx] = Float64.(rand(Distributions.Bernoulli(θ[idx])))
                    if z[idx] > 0.999
                        B -= 1.0
                    end
                    if B < 1e-4
                        break
                    end
                end

                k = 1.0
                for i = 1:length(z)
                    k += z[i] * 2^(i - 1)
                end

                if Int(k) ∉ z_list
                    push!(z_scenario, z)
                    push!(z_list, Int(k))
                    break
                end

                if iter > 10000
                    break
                end
            end
            # println("z_list=", z_list)
        end
    end

    return z_scenario, z_list
end


function compute_objective_value(m, var, pd, z)

    ## set "z"
    for idx in 1:length(z)
        gmd_bus_idx = pd.substations[idx]
        pd.z[gmd_bus_idx] = z[idx]
    end


    m, c1 = gic.add_blockers_model_nlp(m, var, pd)


    ## solve
    JuMP.optimize!(m)

    obj = JuMP.objective_value(m)
    term = JuMP.termination_status(m)

    ## remove  
    m = remove_blockers_model_nlp(m, c1)

    return obj, term
end


function projection_by_truncation(θ)

    θ .= clamp.(θ, 0.0, 1.0)

    return θ
end
function grad_log_pmf(z, θ)

    marginal_pmf = θ .^ z .* (1.0 .- θ) .^ (1.0 .- z)

    grad_marginal_log_prob = []
    for i = 1:length(marginal_pmf)

        ## ones_entries
        if marginal_pmf[i] > 0.99
            push!(grad_marginal_log_prob, 0.0)
            ## valid_entries
        else
            push!(grad_marginal_log_prob, z[i] / θ[i] + (z[i] - 1.0) / (1.0 - θ[i]))
        end
    end

    return grad_marginal_log_prob
end


function find_best_scenario(args, m, var, pd, θ)

    N = args["n_samples"]

    z_scenario, _ = sample_z_scenario(args, N, θ, pd, [])

    obj_list = []
    term_list = []
    for s = 1:length(z_scenario)

        z = z_scenario[s]

        obj, term = compute_objective_value(m, var, pd, z)

        push!(obj_list, obj)
        push!(term_list, term)

    end

    return z_scenario[argmin(obj_list)]
end


function write_output_stochastic(args)

    dir = args["output_dir_name"]

    if isdir(dir) == false
        mkpath(dir)
    end

    path = dir * args["output_file_name"] * "_log"
    outfile = get_new_filename(path, ".txt")
    res_io = open(outfile * ".txt", "w")
    return res_io

end


function log_title_stochastic(res_io)
    @printf(res_io,
        "%10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s \n",
        "iter",
        "nsamples",
        "per[s]",
        "elpsd[s]",
        "stepsize",
        "gradnorm",
        "#θ=1",
        "#θ=0",
        "#θ∈(0,1)",
        "θ"
    )

    flush(res_io)
    @printf(
        "%10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s \n",
        "iter",
        "nsamples",
        "per[s]",
        "elpsd[s]",
        "stepsize",
        "gradnorm",
        "#θ=1",
        "#θ=0",
        "#θ∈(0,1)",
        "θ"
    )

end


function log_iteration_stochastic(res_io, args)

    @printf(res_io,
        "%10d  %10d  %10.4e  %10.4e  %10.4e  %10.4e  %10.4s  %10.4s  %10.4s  %1s     \n",
        args["iter"],
        args["nsamples_exp"],
        args["periter_time"],
        args["elapsed_time"],
        args["η"],
        args["grad_norm"],
        args["cnt_1"],
        args["cnt_0"],
        args["cnt_01"],
        args["θ"]
    )
    flush(res_io)
    @printf(
        "%10d  %10d  %10.4e  %10.4e  %10.4e  %10.4e  %10.4s  %10.4s  %10.4s  %1s     \n",
        args["iter"],
        args["nsamples_exp"],
        args["periter_time"],
        args["elapsed_time"],
        args["η"],
        args["grad_norm"],
        args["cnt_1"],
        args["cnt_0"],
        args["cnt_01"],
        "-"
    )
end