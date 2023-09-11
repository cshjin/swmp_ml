function constraints_power_balance(m, var, pd)
    for bus in pd.Bus
        i = bus.bus_i
        ## var_p
        LE1_1_LHS = 0
        for line in pd.Line
            e = line.line_i
            if i == line.tbus
                LE1_1_LHS = LE1_1_LHS + var.p[e, i]
            end
            if i == line.fbus
                LE1_1_LHS = LE1_1_LHS + var.p[e, i]
            end
        end
        LE1_1_RHS = 0
        for gen in pd.Gen
            k = gen.gen_i
            if i == gen.gbus
                LE1_1_RHS = LE1_1_RHS + var.fp[k]
            end
        end

        LE1_1_RHS = LE1_1_RHS - bus.pd + var.lpp[i] - var.lpm[i] - bus.gs * var.w[i]
        @constraint(m, LE1_1_LHS == LE1_1_RHS)

        ## var_q
        LE1_2_LHS = 0
        for line in pd.Line
            e = line.line_i
            if i == line.tbus
                LE1_2_LHS = LE1_2_LHS + var.q[e, i]
            end
            if i == line.fbus
                LE1_2_LHS = LE1_2_LHS + var.q[e, i]
            end
        end
        LE1_2_RHS = 0
        for gen in pd.Gen
            k = gen.gen_i
            if i == gen.gbus
                LE1_2_RHS = LE1_2_RHS + var.fq[k]
            end
        end
        LE1_2_RHS = LE1_2_RHS - bus.qd + var.lqp[i] - var.lqm[i] + bus.bs * var.w[i] - var.dqloss[i]
        @constraint(m, LE1_2_LHS == LE1_2_RHS)
    end
end

function constraints_power_flow(m, var, pd)
    for line in pd.Line
        e = line.line_i
        i = line.fbus
        j = line.tbus

        @constraint(m, var.p[e, i] == line.g * var.w[i] - line.g * var.wc[e] - line.b * var.ws[e])
        @constraint(m, var.p[e, j] == line.g * var.w[j] - line.g * var.wc[e] + line.b * var.ws[e])
        @constraint(m, var.q[e, i] == -(line.b + line.bc / 2.0) * var.w[i] + line.b * var.wc[e] - line.g * var.ws[e])
        @constraint(m, var.q[e, j] == -(line.b + line.bc / 2.0) * var.w[j] + line.b * var.wc[e] + line.g * var.ws[e])
    end
end

function constraints_ac_polar(m, var, pd)
    for bus in pd.Bus
        i = bus.bus_i
        @constraint(m, var.w[i] == var.v[i]^2)
    end
    for line in pd.Line
        e = line.line_i
        i = line.fbus
        j = line.tbus

        NL1 = @NLexpression(m, var.v[i] * var.v[j] * cos(var.theta[i] - var.theta[j]))
        @NLconstraint(m, var.wc[e] == NL1)
        NL2 = @NLexpression(m, var.v[i] * var.v[j] * sin(var.theta[i] - var.theta[j]))
        @NLconstraint(m, var.ws[e] == NL2)
    end
end

function constraints_soc_polar(m, var, pd)
    for bus in pd.Bus
        i = bus.bus_i
        @constraint(m, var.w[i] >= var.v[i]^2)
        @constraint(m, var.w[i] <= (bus.Vmax + bus.Vmin) * var.v[i] - bus.Vmax * bus.Vmin)
    end
    for line in pd.Line
        e = line.line_i
        i = line.fbus
        j = line.tbus

        @constraint(m, var.wc[e]^2 + var.ws[e]^2 - var.w[i] * var.w[j] <= 0.0)
        @constraint(m, tan(line.angmin) * var.wc[e] <= var.ws[e])
        @constraint(m, var.ws[e] <= tan(line.angmax) * var.wc[e])
    end
end

function constraints_ac_rect(m, var, pd)
    for bus in pd.Bus
        i = bus.bus_i
        @constraint(m, var.w[i] == var.v[i]^2)
        @constraint(m, var.w[i] == var.wrr[i, i] + var.wii[i, i])

        @constraint(m, var.wrr[i, i] == var.vr[i] * var.vr[i])
        @constraint(m, var.wii[i, i] == var.vi[i] * var.vi[i])
    end
    for line in pd.Line
        e = line.line_i
        i = line.fbus
        j = line.tbus
        @constraint(m, var.wc[e] == var.wrr[i, j] + var.wii[i, j])
        @constraint(m, var.ws[e] == var.wri[j, i] - var.wri[i, j])

        @constraint(m, var.wrr[i, j] == var.vr[i] * var.vr[j])
        @constraint(m, var.wii[i, j] == var.vi[i] * var.vi[j])
        @constraint(m, var.wri[i, j] == var.vr[i] * var.vi[j])
        @constraint(m, var.wri[j, i] == var.vr[j] * var.vi[i])

        @constraint(m, tan(line.angmin) * var.wc[e] <= var.ws[e])
        @constraint(m, var.ws[e] <= tan(line.angmax) * var.wc[e])
    end
end

function constraints_soc_rect(m, var, pd)

    for bus in pd.Bus
        i = bus.bus_i
        @constraint(m, var.w[i] >= var.v[i]^2)
        @constraint(m, var.w[i] <= (bus.Vmax + bus.Vmin) * var.v[i] - bus.Vmax * bus.Vmin)
        @constraint(m, var.w[i] == var.wrr[i, i] + var.wii[i, i])
    end
    for line in pd.Line
        e = line.line_i
        i = line.fbus
        j = line.tbus
        @constraint(m, var.wc[e] == var.wrr[i, j] + var.wii[i, j])
        @constraint(m, var.ws[e] == var.wri[j, i] - var.wri[i, j])

        @constraint(m, tan(line.angmin) * var.wc[e] <= var.ws[e])
        @constraint(m, var.ws[e] <= tan(line.angmax) * var.wc[e])

        @constraint(m, (var.wrr[i, j] + var.wii[i, j])^2 + (var.wri[j, i] - var.wri[i, j])^2 +
                       ((var.wrr[i, i] + var.wii[i, i] - var.wrr[j, j] - var.wii[j, j]) / 2.0)^2
                       <=
                       ((var.wrr[i, i] + var.wii[i, i] + var.wrr[j, j] + var.wii[j, j]) / 2.0)^2)
    end

end

function constraints_angle_difference(m, var, pd)
    for line in pd.Line
        i = line.fbus
        j = line.tbus
        @constraint(m, var.theta[i] - var.theta[j] <= line.angmax)
        @constraint(m, var.theta[i] - var.theta[j] >= line.angmin)
    end
end

function constraints_thermal_limit(m, var, pd)
    for line in pd.Line
        e = line.line_i
        i = line.fbus
        j = line.tbus

        @constraint(m, var.p[e, i]^2 + var.q[e, i]^2 <= line.rateA^2)
        @constraint(m, var.p[e, j]^2 + var.q[e, j]^2 <= line.rateA^2)
    end
end

function constraints_gic(m, var, pd)
    for gmdline in pd.GMDLine
        e = gmdline.GMDline_i
        tmp_f = gmdline.fbusd
        tmp_t = gmdline.tbusd
        br_r = gmdline.br_r
        induced_v = pd.mu_E * gmdline.dist_E + pd.mu_N * gmdline.dist_N

        @constraint(m, var.Id[e] == (1.0 / br_r) * (var.vd[tmp_f] - var.vd[tmp_t] + induced_v))
    end
end

function constraints_dqloss(m, var, pd)
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
                    LE = LE + K * var.v[i] * var.Ieff[e] / (3.0 * pd.baseMVA)
                end
            end
        end
        @NLconstraint(m, var.dqloss[i] == LE)
    end
end

function constraints_effective_gic_complementarity(m, var, pd)
    for line in pd.Line
        e = line.line_i

        if line.config == "gwye-delta"
            e_h = line.gmd_br_hi
            # @NLconstraint(m, var.Ieff[e] == abs( var.Id[e_h] ) )
            @constraint(m, var.Ieff_plus[e] * var.Ieff_minus[e] <= 0.0)
            @constraint(m, var.Ieff[e] == var.Ieff_plus[e] + var.Ieff_minus[e])
            @constraint(m, var.Id[e_h] == var.Ieff_plus[e] - var.Ieff_minus[e])

        end
        if line.config == "gwye-gwye"
            e_h = line.gmd_br_hi
            e_l = line.gmd_br_lo

            # @NLconstraint(m, var.Ieff[e] == abs((pd.turn_ratio[e] * var.Id[e_h] + var.Id[e_l]) / pd.turn_ratio[e]) )
            @NLconstraint(m, (pd.turn_ratio[e] * var.Id[e_h] + var.Id[e_l]) / pd.turn_ratio[e] == var.Ieff_plus[e] - var.Ieff_minus[e])

            @constraint(m, var.Ieff_plus[e] * var.Ieff_minus[e] <= 0.0)
            @constraint(m, var.Ieff[e] == var.Ieff_plus[e] + var.Ieff_minus[e])
        end
        if line.config == "gwye-gwye-auto"
            e_s = line.gmd_br_series
            e_c = line.gmd_br_common

            # @NLconstraint(m, var.Ieff[e] == abs( (pd.turn_ratio[e] * var.Id[e_s] + var.Id[e_c]) / (pd.turn_ratio[e] + 1.0) ) )
            @NLconstraint(m, (pd.turn_ratio[e] * var.Id[e_s] + var.Id[e_c]) / (pd.turn_ratio[e] + 1.0) == var.Ieff_plus[e] - var.Ieff_minus[e])
            @constraint(m, var.Ieff_plus[e] * var.Ieff_minus[e] <= 0.0)
            @constraint(m, var.Ieff[e] == var.Ieff_plus[e] + var.Ieff_minus[e])
        end
    end
end

function constraints_blockers(args, m, var, pd)
    LE = 0
    for gmdbus in pd.GMDBus
        i = gmdbus.GMDbus_i
        if gmdbus.g_gnd > 0.0 ## positive for substations
            LE += var.z[i]
        end
    end
    @constraint(m, LE <= args["budget"])
end


################################################################################################
function constraints_gic_balance_with_blockers_var(m, var, pd)
    ## GIC constraint
    for gmdbus in pd.GMDBus
        i = gmdbus.GMDbus_i
        LE2 = 0
        for gmdline in pd.GMDLine
            e = gmdline.GMDline_i
            if gmdbus.GMDbus_i == gmdline.tbusd
                LE2 = LE2 + var.Id[e]
            end
            if gmdbus.GMDbus_i == gmdline.fbusd
                LE2 = LE2 - var.Id[e]
            end
        end

        if gmdbus.g_gnd > 0.0 ## positive for substations
            @NLconstraint(m, LE2 == gmdbus.g_gnd * var.vd[i] * (1.0 - var.z[i]))
        else
            @constraint(m, LE2 == 0.0)
        end
    end
end

function constraints_gic_balance_with_blockers_fix(m, var, pd)
    c1 = Dict()
    for gmdbus in pd.GMDBus
        i = gmdbus.GMDbus_i

        LE2 = 0
        for gmdline in pd.GMDLine
            e = gmdline.GMDline_i
            if gmdbus.GMDbus_i == gmdline.tbusd
                LE2 = LE2 + var.Id[e]
            end
            if gmdbus.GMDbus_i == gmdline.fbusd
                LE2 = LE2 - var.Id[e]
            end
        end
        if gmdbus.g_gnd > 0.0 ## positive for substations
            c1[i] = @NLconstraint(m, LE2 == gmdbus.g_gnd * var.vd[i] * (1.0 - pd.z[i]))
        else
            c1[i] = @constraint(m, LE2 == 0.0)
        end
    end
    return m, c1
end

function constraints_gic_balance_with_blockers_trust(m, var, pd)
    c1 = Dict()
    pd.trust_consensus = Dict()
    for gmdbus in pd.GMDBus
        i = gmdbus.GMDbus_i

        LE2 = 0
        for gmdline in pd.GMDLine
            e = gmdline.GMDline_i
            if gmdbus.GMDbus_i == gmdline.tbusd
                LE2 = LE2 + var.Id[e]
            end
            if gmdbus.GMDbus_i == gmdline.fbusd
                LE2 = LE2 - var.Id[e]
            end
        end
        if gmdbus.g_gnd > 0.0 ## positive for substations
            c1[i] = @NLconstraint(m, LE2 == gmdbus.g_gnd * var.vd[i] * (1.0 - pd.z[i]))
            pd.trust_consensus[i] = @constraint(m, var.z[i] == pd.z[i])
        else
            c1[i] = @constraint(m, LE2 == 0.0)
        end

    end

    return m, c1
end


################################################################################################
# function constraints_effective_gic_var(m, var, pd)    
#     for line in pd.Line
#         e = line.line_i  
#         if line.config == "gwye-gwye"
#             e_h = line.gmd_br_hi
#             e_l = line.gmd_br_lo
#             sub_idx =  pd.gmd_line_substation[(e_h,e_l)]    
#             @NLconstraint(m, (1.0-var.z[sub_idx]) * (pd.turn_ratio[e] * var.Id[e_h] + var.Id[e_l]) / pd.turn_ratio[e] == var.Ieff_plus[e] - var.Ieff_minus[e] )            
#         end
#         if line.config == "gwye-gwye-auto"
#             e_s = line.gmd_br_series
#             e_c = line.gmd_br_common
#             sub_idx =  pd.gmd_line_substation[(e_s,e_c)] 
#             @NLconstraint(m, (1.0-var.z[sub_idx]) * (pd.turn_ratio[e] * var.Id[e_s] + var.Id[e_c]) / (pd.turn_ratio[e] + 1.0) == var.Ieff_plus[e] - var.Ieff_minus[e] )            
#         end
#     end    
# end
# function constraints_effective_gic_fix(m, var, pd)    
#     c2 = Dict()
#     for line in pd.Line
#         e = line.line_i  
#         if line.config == "gwye-gwye"
#             e_h = line.gmd_br_hi
#             e_l = line.gmd_br_lo
#             sub_idx =  pd.gmd_line_substation[(e_h,e_l)]    
#             c2[e] = @NLconstraint(m, (1.0-pd.z[sub_idx]) * (pd.turn_ratio[e] * var.Id[e_h] + var.Id[e_l]) / pd.turn_ratio[e] == var.Ieff_plus[e] - var.Ieff_minus[e] )            
#         end
#         if line.config == "gwye-gwye-auto"
#             e_s = line.gmd_br_series
#             e_c = line.gmd_br_common
#             sub_idx =  pd.gmd_line_substation[(e_s,e_c)] 
#             c2[e] = @NLconstraint(m, (1.0-pd.z[sub_idx]) * (pd.turn_ratio[e] * var.Id[e_s] + var.Id[e_c]) / (pd.turn_ratio[e] + 1.0) == var.Ieff_plus[e] - var.Ieff_minus[e] )            
#         end
#     end    
#     return m, c2
# end