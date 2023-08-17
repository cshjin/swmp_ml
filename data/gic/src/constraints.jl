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

####################start##########################
function constraints_nonlinear_polar(m, var, pd)
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

function constraints_nonlinear_rect(m, var, pd)
    for bus_1 in pd.Bus
        i = bus_1.bus_i
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

function constraints_soc_rect(m, var, pd)

    for bus_1 in pd.Bus
        i = bus_1.bus_i
        @constraint(m, var.w[i] == var.wrr[i, i] + var.wii[i, i])
    end
    for line in pd.Line
        e = line.line_i
        i = line.fbus
        j = line.tbus
        @constraint(m, var.wc[e] == var.wrr[i, j] + var.wii[i, j])
        @constraint(m, var.ws[e] == var.wri[j, i] - var.wri[i, j])

        @constraint(m, (var.wrr[i, j] + var.wii[i, j])^2 + (var.wri[j, i] - var.wri[i, j])^2 +
                       ((var.wrr[i, i] + var.wii[i, i] - var.wrr[j, j] - var.wii[j, j]) / 2.0)^2
                       <=
                       ((var.wrr[i, i] + var.wii[i, i] + var.wrr[j, j] + var.wii[j, j]) / 2.0)^2)
    end

end
####################end############################

function constraints_angle_difference_polar(m, var, pd)
    for line in pd.Line
        i = line.fbus
        j = line.tbus
        @constraint(m, var.theta[i] - var.theta[j] <= line.angmax)
        @constraint(m, var.theta[i] - var.theta[j] >= line.angmin)
    end
end

function constraints_angle_difference_rect(m, var, pd)
    for line in pd.Line
        e = line.line_i
        @constraint(m, var.ws[e] <= tan(line.angmax) * var.wc[e])
        @constraint(m, tan(line.angmin) * var.wc[e] <= var.ws[e])
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


####################start##########################
function constraints_gic_balance_with_blockers_nonlinear(m, var, pd)

    if isempty(pd.z) == true
        LE = 0
        for gmdbus in pd.GMDBus
            i = gmdbus.GMDbus_i
            LE += var.z[i]
        end
        @constraint(m, LE <= pd.tot_num_blockers)
    end


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

        if isempty(pd.z) == true
            @NLconstraint(m, LE2 == gmdbus.g_gnd * var.vd[i] * (1.0 - var.z[i]))
        else
            @constraint(m, LE2 == gmdbus.g_gnd * var.vd[i] * (1.0 - pd.z[i]))
        end
    end


end

function constraints_gic_balance_with_blockers_mccorkmick(m, var, pd)
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
        constraints_mccormick(m, var.v_mc[i], var.vd[i], var.z[i], 0.0, pd.vdmax, 0.0, 1.0)
        @NLconstraint(m, LE2 == gmdbus.g_gnd * (var.vd[i] - var.v_mc[i]))
    end
end
####################end############################


####################start##########################
function constraints_dqloss_nonlinear_polar(m, var, pd)
    for bus in pd.Bus
        i = bus.bus_i
        LE = 0
        for line in pd.Line
            e = line.line_i
            if line.type == "xf"
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

function constraints_dqloss_nonlinear_rect(m, var, pd)
    for bus in pd.Bus
        i = bus.bus_i
        temp_list = []
        for line in pd.Line
            e = line.line_i
            if line.type == "xf"
                if i == line.fbus || i == line.tbus
                    baseKV = pd.Bus[pd.Bus_index2id[line.hi_bus]].baseKV
                    ibase = pd.baseMVA * 1000.0 * sqrt(2.0) / (baseKV * sqrt(3.0))
                    K = line.gmd_k * pd.baseMVA / ibase
                    NL1 = @NLexpression(m, K * sqrt(var.w[i]) * var.Ieff[e] / (3.0 * pd.baseMVA))
                    push!(temp_list, NL1)
                end
            end
        end
        NL2 = @NLexpression(m, sum(k for k in temp_list))
        @NLconstraint(m, var.dqloss[i] == NL2)
    end
end

function constraints_dqloss_mccormick_polar(m, var, pd)
    for bus in pd.Bus
        i = bus.bus_i
        LE4 = 0
        for line in pd.Line
            e = line.line_i
            if line.type == "xf"
                if i == line.fbus || i == line.tbus
                    constraints_mccormick(m, var.u_mc[i, e], var.v[i], var.Ieff[e], bus.Vmin, bus.Vmax, 0.0, pd.Immax)

                    baseKV = pd.Bus[pd.Bus_index2id[line.hi_bus]].baseKV
                    ibase = pd.baseMVA * 1000.0 * sqrt(2.0) / (baseKV * sqrt(3.0))
                    K = line.gmd_k * pd.baseMVA / ibase
                    LE4 = LE4 + K * var.u_mc[i, e] / (3.0 * pd.baseMVA)
                end
            end
        end
        @constraint(m, var.dqloss[i] == LE4)
    end
end
####################end############################


####################start##########################
function constraints_effective_gic_absolute(m, var, pd)
    for line in pd.Line
        if line.type == "xf"
            e = line.line_i

            if line.config == "gwye-delta"
                e_h = line.gmd_br_hi
                @NLconstraint(m, var.Ieff[e] == abs(var.Id[e_h]))
            end
            if line.config == "gwye-gwye"
                e_h = line.gmd_br_hi
                e_l = line.gmd_br_lo

                @NLconstraint(m, var.Ieff[e] == abs((pd.turn_ratio[e] * var.Id[e_h] + var.Id[e_l]) / pd.turn_ratio[e]))
            end
            if line.config == "gwye-gwye-auto"
                e_s = line.gmd_br_series
                e_c = line.gmd_br_common

                @NLconstraint(m, var.Ieff[e] == abs((pd.turn_ratio[e] * var.Id[e_s] + var.Id[e_c]) / (pd.turn_ratio[e] + 1.0)))
            end
        end
    end
end
function constraints_effective_gic_absolute_reform(m, var, pd)
    for line in pd.Line
        if line.type == "xf"
            e = line.line_i

            if line.config == "gwye-delta"
                e_h = line.gmd_br_hi
                # @NLconstraint(m, var.Ieff[e] == abs( var.Id[e_h] ) )
                @constraint(m, var.Ieff[e] == var.Ieff_plus[e] + var.Ieff_minus[e])
                @constraint(m, var.Id[e_h] == var.Ieff_plus[e] - var.Ieff_minus[e])
                @constraint(m, var.Ieff_plus[e] * var.Ieff_minus[e] <= 0.0)


            end
            if line.config == "gwye-gwye"
                e_h = line.gmd_br_hi
                e_l = line.gmd_br_lo

                # @NLconstraint(m, var.Ieff[e] == abs((pd.turn_ratio[e] * var.Id[e_h] + var.Id[e_l]) / pd.turn_ratio[e]) )
                @constraint(m, var.Ieff[e] == var.Ieff_plus[e] + var.Ieff_minus[e])
                @constraint(m, (pd.turn_ratio[e] * var.Id[e_h] + var.Id[e_l]) / pd.turn_ratio[e] == var.Ieff_plus[e] - var.Ieff_minus[e])
                @constraint(m, var.Ieff_plus[e] * var.Ieff_minus[e] <= 0.0)
            end
            if line.config == "gwye-gwye-auto"
                e_s = line.gmd_br_series
                e_c = line.gmd_br_common

                # @NLconstraint(m, var.Ieff[e] == abs( (pd.turn_ratio[e] * var.Id[e_s] + var.Id[e_c]) / (pd.turn_ratio[e] + 1.0) ) )
                @constraint(m, var.Ieff[e] == var.Ieff_plus[e] + var.Ieff_minus[e])
                @constraint(m, (pd.turn_ratio[e] * var.Id[e_s] + var.Id[e_c]) / (pd.turn_ratio[e] + 1.0) == var.Ieff_plus[e] - var.Ieff_minus[e])
                @constraint(m, var.Ieff_plus[e] * var.Ieff_minus[e] <= 0.0)
            end
        end
    end
end

function constraints_effective_gic_relaxation(m, var, pd)
    for line in pd.Line
        if line.type == "xf"
            e = line.line_i
            if line.config == "gwye-delta"
                e_h = line.gmd_br_hi
                @constraint(m, var.Ieff[e] >= var.Id[e_h])
                @constraint(m, var.Ieff[e] >= -var.Id[e_h])
            end
            if line.config == "gwye-gwye"
                e_h = line.gmd_br_hi
                e_l = line.gmd_br_lo
                LE3_1 = (pd.turn_ratio[e] * var.Id[e_h] + var.Id[e_l]) / pd.turn_ratio[e]
                @constraint(m, var.Ieff[e] >= LE3_1)
                @constraint(m, var.Ieff[e] >= -LE3_1)
            end
            if line.config == "gwye-gwye-auto"
                e_s = line.gmd_br_series
                e_c = line.gmd_br_common
                LE3_2 = (pd.turn_ratio[e] * var.Id[e_s] + var.Id[e_c]) / (pd.turn_ratio[e] + 1.0)
                @constraint(m, var.Ieff[e] >= LE3_2)
                @constraint(m, var.Ieff[e] >= -LE3_2)
            end
        end
    end
end
####################end############################

function constraints_mccormick(m, z, x, y, xmin, xmax, ymin, ymax)
    @constraint(m, z >= xmin * y + ymin * x - xmin * ymin)
    @constraint(m, z >= xmax * y + ymax * x - xmax * ymax)
    @constraint(m, -z >= -xmin * y - ymax * x + xmin * ymax)
    @constraint(m, -z >= -xmax * y - ymin * x + xmax * ymin)
end