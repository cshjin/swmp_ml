using Distributions
using LinearAlgebra
function projected_sgd(args, m, var, pd)
    
    ## Initialization   
    gmdbus_idx = []
    θ = []
    for gmdbus in pd.GMDBus
        i = gmdbus.GMDbus_i
        push!(gmdbus_idx, i)
        push!(θ, 0.5) 
    end
 

    N = args["num_sample"]
    obj_dict=Dict{}
    term_dict=Dict{}
    for t = 1:3
        println("t=",t, " θ=",θ)

        η = args["lr"] / t
    
        z_scenario = gic.sample_z_scenario(N, θ, pd)
        
        stime = time()
        obj_list=[]
        p_grad_list=[]
        term_list=[] 
        grad = 0.0
        for s = 1:N         

            z = z_scenario[s]

            ## 1)
            m = gic.compute_objective_value(args, m, var, pd, z)  
            obj = JuMP.objective_value(m)
            term = JuMP.termination_status(m)
            println("Termination status=", term)
            push!(obj_list, obj)
            push!(term_list, term)

            ## 2) 
            p_grad = z ./ θ .+ (z .- 1) ./ (1.0 .- θ ) 
            push!(p_grad_list, p_grad)  
            println(s, "   ", "   ",  obj, "   ", p_grad, "   ")

            grad = grad .+ obj * p_grad            
        end                 
        grad = grad ./ N
        
        println("grad=", grad)


        θ = θ - η .* grad        
        θ = gic.projection_by_truncation(θ)
        grad_project = gic.projection_by_truncation(grad)
        
        println("grad_norm=",LinearAlgebra.norm(grad_project))
  
        
        args["elapsed_time"] = time() - stime
        # println("elapsed_time=", args["elapsed_time"])
        
        # println("obj_list=",obj_list)
        # println("p_grad_list=",p_grad_list)
        # grad = obj_list .* p_grad_list
        # println("grad=",grad)
        # stop
        


        # stop
        # obj_dict[t]=obj_list
        # term_dict[t]=term_list

         
    end    

end
 
function projection_by_truncation(θ)

    θ .= clamp.(θ, 0.0, 1.0)
 
    return θ
end


function compute_objective_value(args, m, var, pd, z_scenario)
    ## set "z"
    for gmdbus in pd.GMDBus
        i = gmdbus.GMDbus_i
        pd.z[i] = z_scenario[i]
    end

    
    ## model
    if args["model"] == "ac_polar"
        m = gic.construct_gic_blockers_ac_polar_model(m, var, pd)
    end
    if args["model"] == "ac_rect"
        m = gic.construct_gic_blockers_ac_rect_model(m, var, pd)
    end
    if args["model"] == "soc_polar"
        m = gic.construct_gic_blockers_soc_polar_model(m, var, pd)
    end
    if args["model"] == "soc_rect"
        m = gic.construct_gic_blockers_soc_rect_model(m, var, pd)
    end

    ## additional objective function value
    obj_curr = JuMP.objective_function(m)     
    z_sum = sum(values(pd.z))    
    obj_updt =  args["penalty"]*max(0, z_sum - args["tot_num_blockers"])^2    
    set_objective_function(m, obj_curr + obj_updt)
     
    ## solve
    JuMP.optimize!(m)
    
    return m
end


function sample_z_scenario(N, θ, pd)
    z_scenario = []
    
    for s = 1:N        
        temp = []
        for gmdbus in pd.GMDBus
            i = gmdbus.GMDbus_i             
            push!(temp, Float64.(rand(Distributions.Bernoulli(θ[i]))) )
        end 
        push!(z_scenario, temp)
    end
    
    return z_scenario
end