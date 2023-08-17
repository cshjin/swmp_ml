function variables_generators(m, var, pd)
    ## generators in AC network
    var.fp=Dict();var.fq=Dict();
    for gen in pd.Gen
        k = gen.gen_i
        var.fp[k] = @variable(m, lower_bound = gen.Pmin , upper_bound = gen.Pmax, base_name="fp[$(k)]")
        var.fq[k] = @variable(m, lower_bound = gen.Qmin , upper_bound = gen.Qmax, base_name="fq[$(k)]") 
    end    
end

function variables_buses_polar(m, var, pd)
    ## buses in AC network
    var.v=Dict(); var.w=Dict(); var.theta=Dict(); 
    var.lpp=Dict(); var.lpm=Dict(); var.lqp=Dict(); var.lqm=Dict(); var.dqloss=Dict();
    for bus in pd.Bus
        i = bus.bus_i
        var.v[i] = @variable(m, lower_bound = bus.Vmin, upper_bound = bus.Vmax, base_name="v[$(i)]")
        var.w[i] = @variable(m, lower_bound = bus.wmin, upper_bound = bus.wmax, base_name="w[$(i)]")
        var.theta[i] = @variable(m, base_name="theta[$(i)]")
        var.lpp[i] = @variable(m, lower_bound = 0.0,  base_name="lpp[$(i)]")
        var.lpm[i] = @variable(m, lower_bound = 0.0,  base_name="lpm[$(i)]")
        var.lqp[i] = @variable(m, lower_bound = 0.0,  base_name="lqp[$(i)]")
        var.lqm[i] = @variable(m, lower_bound = 0.0,  base_name="lqm[$(i)]")
        var.dqloss[i] = @variable(m, lower_bound = 0.0,  base_name="dqloss[$(i)]")
    end
end

function variables_buses_rect(m, var, pd)
    ## buses in AC network
    var.wrr=Dict(); var.wii=Dict();   
    var.w=Dict(); var.vr=Dict(); var.vi=Dict();   
    var.lpp=Dict(); var.lpm=Dict(); var.lqp=Dict(); var.lqm=Dict(); var.dqloss=Dict();
    for bus_1 in pd.Bus
        i = bus_1.bus_i
        var.wrr[i,i] = @variable(m, lower_bound = -bus_1.Vmax*bus_1.Vmax, upper_bound = bus_1.Vmax*bus_1.Vmax, base_name="wrr[$(i), $(i)]")
        var.wii[i,i] = @variable(m, lower_bound = -bus_1.Vmax*bus_1.Vmax, upper_bound = bus_1.Vmax*bus_1.Vmax,base_name="wii[$(i), $(i)]")
        var.w[i] = @variable(m, lower_bound = bus_1.wmin, upper_bound = bus_1.wmax, base_name="w[$(i)]")
         
        var.vr[i] = @variable(m, lower_bound = -bus_1.Vmax+0.000000000000001, upper_bound = bus_1.Vmax, base_name="vr[$(i)]")
        var.vi[i] = @variable(m, lower_bound = -bus_1.Vmax, upper_bound = bus_1.Vmax,base_name="vi[$(i)]")        
        var.lpp[i] = @variable(m, lower_bound = 0.0,  base_name="lpp[$(i)]")
        var.lpm[i] = @variable(m, lower_bound = 0.0,  base_name="lpm[$(i)]")
        var.lqp[i] = @variable(m, lower_bound = 0.0,  base_name="lqp[$(i)]")
        var.lqm[i] = @variable(m, lower_bound = 0.0,  base_name="lqm[$(i)]")
        var.dqloss[i] = @variable(m, lower_bound = 0.0,  base_name="dqloss[$(i)]")
    end    
end

function variables_lines_rect(m, var, pd)
    ## lines in AC network
    var.wri=Dict(); 
    var.p=Dict(); var.q=Dict(); var.wc=Dict(); var.ws=Dict();
    var.Ieff=Dict(); var.Ieff_plus=Dict(); var.Ieff_minus=Dict()
    for line in pd.Line
        e = line.line_i
        i = line.fbus
        j = line.tbus

        var.p[e,i] = @variable(m, base_name="p[$(e),$(i)]")
        var.p[e,j] = @variable(m, base_name="p[$(e),$(j)]")

        var.q[e,i] = @variable(m, base_name="q[$(e),$(i)]")
        var.q[e,j] = @variable(m, base_name="q[$(e),$(j)]")

        var.wc[e] = @variable(m, lower_bound = line.wcmin, upper_bound = line.wcmax, base_name="wc[$(e)]")
        var.ws[e] = @variable(m, lower_bound = line.wsmin, upper_bound = line.wsmax, base_name="ws[$(e)]") 

        var.Ieff[e] = @variable(m, lower_bound = 0.0, upper_bound = pd.Immax, base_name="Ieff[$(e)]")
        var.Ieff_plus[e] = @variable(m, lower_bound = 0.0, upper_bound = pd.Immax, base_name="Ieff[$(e)]")
        var.Ieff_minus[e] = @variable(m, lower_bound = 0.0, upper_bound = pd.Immax, base_name="Ieff[$(e)]")

        vmax_i = pd.Bus[ pd.Bus_index2id[i] ].Vmax
        vmax_j = pd.Bus[ pd.Bus_index2id[j] ].Vmax
        
        var.wrr[i,j] = @variable(m, lower_bound = - vmax_i * vmax_j, upper_bound = vmax_i * vmax_j, base_name="wrr[$(i), $(j)]")        
        var.wii[i,j] = @variable(m, lower_bound = - vmax_i * vmax_j, upper_bound = vmax_i * vmax_j, base_name="wii[$(i), $(j)]")        
        var.wri[i,j] = @variable(m, lower_bound = - vmax_i * vmax_j, upper_bound = vmax_i * vmax_j, base_name="wri[$(i), $(j)]")
        var.wri[j,i] = @variable(m, lower_bound = - vmax_i * vmax_j, upper_bound = vmax_i * vmax_j, base_name="wri[$(j), $(i)]")                
    end
end

function variables_lines_polar(m, var, pd)
    ## lines in AC network
    var.p=Dict(); var.q=Dict(); var.wc=Dict(); var.ws=Dict();
    var.Ieff=Dict(); var.Ieff_plus=Dict(); var.Ieff_minus=Dict()
    for line in pd.Line
        e = line.line_i
        i = line.fbus
        j = line.tbus

        var.p[e,i] = @variable(m, base_name="p[$(e),$(i)]")
        var.p[e,j] = @variable(m, base_name="p[$(e),$(j)]")

        var.q[e,i] = @variable(m, base_name="q[$(e),$(i)]")
        var.q[e,j] = @variable(m, base_name="q[$(e),$(j)]")

        var.wc[e] = @variable(m, lower_bound = line.wcmin, upper_bound = line.wcmax, base_name="wc[$(e)]")
        var.ws[e] = @variable(m, lower_bound = line.wsmin, upper_bound = line.wsmax, base_name="ws[$(e)]") 

        var.Ieff[e] = @variable(m, lower_bound = 0.0, upper_bound = pd.Immax, base_name="Ieff[$(e)]")
        var.Ieff_plus[e] = @variable(m, lower_bound = 0.0, upper_bound = pd.Immax, base_name="Ieff[$(e)]")
        var.Ieff_minus[e] = @variable(m, lower_bound = 0.0, upper_bound = pd.Immax, base_name="Ieff[$(e)]")
    end
end

function variables_buses_dc(m, var, pd)
    var.vd=Dict(); 
    for gmdbus in pd.GMDBus
        i = gmdbus.GMDbus_i
        var.vd[i] = @variable(m, base_name="vd[$(i)]")              
    end    

    if isempty(pd.z)==true
        var.z=Dict();
        for gmdbus in pd.GMDBus
            i = gmdbus.GMDbus_i 
            var.z[i] = @variable(m, base_name="z[$(i)]", binary=true) 
        end           
    end


end

function variables_lines_dc(m, var, pd)
    var.Id=Dict();
    for gmdline in pd.GMDLine
        e = gmdline.GMDline_i
        var.Id[e] = @variable(m, base_name="Id[$(e)]")        
    end
end

function variables_mccormick(m, var, pd)
    var.u_mc=Dict()
    for bus in pd.Bus
        i = bus.bus_i
        for line in pd.Line
            e = line.line_i
            if line.type == "xf"
                if i == line.fbus || i == line.tbus
                    var.u_mc[i,e] = @variable(m, base_name="u_mc[$(i),$(e)]")
                end
            end
        end
    end
    var.v_mc=Dict()
    for gmdbus in pd.GMDBus
        i = gmdbus.GMDbus_i        
        var.v_mc[i] = @variable(m, base_name="v_mc[$(i)]") 
    end   
end