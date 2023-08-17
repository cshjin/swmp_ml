include("variables.jl")
include("objective.jl")
include("constraints.jl")
 
 
function construct_gic_blockers_ac_polar_model(m, var, pd)

    variables_generators(m, var, pd)
    variables_buses_polar(m, var, pd) ##
    variables_lines_polar(m, var, pd) ##
    variables_buses_dc(m, var, pd)
    variables_lines_dc(m, var, pd)
    
    construct_objective(m, var, pd)
    
    constraints_power_balance(m, var, pd)
    constraints_power_flow(m, var, pd)    
    constraints_angle_difference_polar(m, var, pd) ##
    constraints_thermal_limit(m, var, pd)
    
    constraints_nonlinear_polar(m, var, pd) ##

    constraints_gic(m, var, pd)    
    
    constraints_gic_balance_with_blockers_nonlinear(m, var, pd)
    constraints_dqloss_nonlinear_polar(m, var, pd) ##
    constraints_effective_gic_absolute_reform(m, var, pd) 
     
    return m
end

function construct_gic_blockers_ac_rect_model(m, var, pd)

    variables_generators(m, var, pd)
    variables_buses_rect(m, var, pd) ##
    variables_lines_rect(m, var, pd) ##
    variables_buses_dc(m, var, pd)
    variables_lines_dc(m, var, pd)
    
    construct_objective(m, var, pd)
    
    constraints_power_balance(m, var, pd)
    constraints_power_flow(m, var, pd)    
    constraints_angle_difference_rect(m, var, pd) ##
    constraints_thermal_limit(m, var, pd)
    
    constraints_nonlinear_rect(m, var, pd) ##
    

    constraints_gic(m, var, pd)        
    constraints_gic_balance_with_blockers_nonlinear(m, var, pd)
    constraints_dqloss_nonlinear_rect(m, var, pd) ##
    constraints_effective_gic_absolute_reform(m, var, pd) 
     
    return m
end
 
function construct_gic_blockers_soc_polar_model(m, var, pd)

    variables_generators(m, var, pd)
    variables_buses_polar(m, var, pd) ##
    variables_lines_polar(m, var, pd) ##
    variables_buses_dc(m, var, pd)
    variables_lines_dc(m, var, pd)
    
    construct_objective(m, var, pd)
    
    constraints_power_balance(m, var, pd)
    constraints_power_flow(m, var, pd)    
    # constraints_angle_difference_polar(m, var, pd) ## included in constraints_soc_polar
    constraints_thermal_limit(m, var, pd)
    
    constraints_soc_polar(m, var, pd) ##

    variables_mccormick(m, var, pd)

    constraints_gic(m, var, pd)        
    
    ## nonlinear GIC model
    constraints_gic_balance_with_blockers_nonlinear(m, var, pd)
    constraints_dqloss_nonlinear_polar(m, var, pd) ##
    constraints_effective_gic_absolute_reform(m, var, pd) 

    ## convex relaxation of the GIC model
    # constraints_gic_balance_with_blockers_mccorkmick(m, var, pd)
    # constraints_dqloss_mccormick_polar(m, var, pd) ##
    # constraints_effective_gic_relaxation(m, var, pd) 
     
    return m
end

function construct_gic_blockers_soc_rect_model(m, var, pd)

    variables_generators(m, var, pd)
    variables_buses_rect(m, var, pd) ##
    variables_lines_rect(m, var, pd) ##
    variables_buses_dc(m, var, pd)
    variables_lines_dc(m, var, pd)
    
    construct_objective(m, var, pd)
    
    constraints_power_balance(m, var, pd)
    constraints_power_flow(m, var, pd)    
    constraints_angle_difference_rect(m, var, pd) ##
    constraints_thermal_limit(m, var, pd)
    
    constraints_soc_rect(m, var, pd) ##

    # variables_mccormick(m, var, pd)

    constraints_gic(m, var, pd)    
    
     
    ## nonlinear GIC model
    constraints_gic_balance_with_blockers_nonlinear(m, var, pd)
    constraints_dqloss_nonlinear_rect(m, var, pd) ##
    constraints_effective_gic_absolute_reform(m, var, pd) 

    ## convex relaxation of the GIC model
    # constraints_gic_balance_with_blockers_mccorkmick(m, var, pd)
    # constraints_dqloss_nonlinear_rect(m, var, pd) ##
    # constraints_effective_gic_relaxation(m, var, pd) 

    return m
end 