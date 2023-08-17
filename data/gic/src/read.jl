function read_data(args, pd)
    ## initial
    pd.z = Dict()

    ## parameter settings
    pd.tot_num_blockers = args["tot_num_blockers"]
    pd.penalty1 = 50
    pd.baseMVA = 100
    pd.vdmax = 7e5 ## 7e5        
    pd.mu_E = args["efield_mag"] * cos(args["efield_dir"] * pi / 180)
    pd.mu_N = args["efield_mag"] * sin(args["efield_dir"] * pi / 180)

    ## read
    Inputfilename = "./data/excel/$(args["network"]).xlsx"

    xf = XLSX.readxlsx(Inputfilename)
    Bus_input = xf["Bus"]
    Gen_input = xf["Generator"]
    Line_input = xf["Line"]
    GMDBus_input = xf["GMDBus"]
    GMDLine_input = xf["GMDLine"]

    pd.Bus = []
    pd.Bus_index2id = Dict()
    tmpcnt = 0
    for row in XLSX.eachrow(Bus_input)
        rn = XLSX.row_number(row) # row number
        if rn == 1
            continue
        else
            bus = BusData()
            bus.bus_i = row[1]
            bus.pd = row[3] / pd.baseMVA
            bus.qd = row[4] / pd.baseMVA
            bus.gs = row[5] / pd.baseMVA
            bus.bs = row[6] / pd.baseMVA
            bus.baseKV = row[10]
            bus.Vmax = row[12]
            bus.Vmin = row[13]
            bus.lat = row[14]
            bus.lon = row[15]
            tmpcnt = tmpcnt + 1
            bus.bus_ID = tmpcnt
            pd.Bus_index2id[bus.bus_i] = bus.bus_ID

        end
        push!(pd.Bus, bus)
    end

    pd.Gen = []
    tmpcnt = 0
    for row in XLSX.eachrow(Gen_input)
        rn = XLSX.row_number(row) # row number
        if rn == 1
            continue
        else
            gen = GenData()
            gen.gen_i = row[1]
            gen.gbus = row[2]
            gen.Qmax = row[5] / pd.baseMVA
            gen.Qmin = row[6] / pd.baseMVA
            gen.Pmax = row[10] / pd.baseMVA
            gen.Pmin = row[11] / pd.baseMVA
            gen.cF2 = row[26]
            gen.cF1 = row[27]
            gen.cF0 = row[28]
            tmpcnt = tmpcnt + 1
            gen.gen_ID = tmpcnt
        end
        push!(pd.Gen, gen)
    end

    pd.Line = []
    tmpcnt = 0
    for row in XLSX.eachrow(Line_input)
        rn = XLSX.row_number(row) # row number
        if rn == 1
            continue
        else
            line = LineData()
            line.line_i = row[1]
            line.fbus = row[2]
            line.tbus = row[3]
            line.r = row[4]
            line.x = row[5]
            line.bc = row[6]
            line.rateA = row[7] / pd.baseMVA
            line.angmin = row[13] / 180 * pi
            line.angmax = row[14] / 180 * pi
            line.hi_bus = row[15]
            line.gmd_br_hi = row[17]
            line.gmd_br_lo = row[18]
            line.gmd_k = row[19]
            line.gmd_br_series = row[20]
            line.gmd_br_common = row[21]
            line.type = row[23]
            line.config = row[24]
            tmpcnt = tmpcnt + 1
            line.line_ID = tmpcnt

        end
        push!(pd.Line, line)
    end

    pd.GMDBus = []
    tmpcnt = 0
    for row in XLSX.eachrow(GMDBus_input)
        rn = XLSX.row_number(row) # row number
        if rn == 1
            continue
        else
            gmdbus = GMDBusData()
            gmdbus.GMDbus_i = row[1]
            gmdbus.parent_bus = row[2]
            gmdbus.g_gnd = row[3]
            tmpcnt = tmpcnt + 1
            gmdbus.GMDbus_ID = tmpcnt
        end
        push!(pd.GMDBus, gmdbus)
    end


    pd.GMDLine = []
    tmpcnt = 0
    for row in XLSX.eachrow(GMDLine_input)
        rn = XLSX.row_number(row) # row number
        if rn == 1
            continue
        else
            gmdline = GMDLineData()
            gmdline.GMDline_i = row[1]
            gmdline.fbusd = row[2]
            gmdline.tbusd = row[3]
            gmdline.parent_branch = row[4]
            gmdline.br_r = row[6]
            gmdline.dist_E = row[10]
            gmdline.dist_N = row[11]
            tmpcnt = tmpcnt + 1
            gmdline.GMDline_ID = tmpcnt

        end
        push!(pd.GMDLine, gmdline)
    end

    # for bus in pd.Bus
    #     println("bus_ID=", bus.bus_ID, ", bus_i=",bus.bus_i, ", pd=",bus.pd, ", qd=",bus.qd, ", gs=",bus.gs,
    #     ", bs=",bus.bs, ", baseKV=",bus.baseKV, ", Vmax=",bus.Vmax, ", Vmin=",bus.Vmin, ", lat=",bus.lat, ", lon=",bus.lon)
    # end 
    # for gen in pd.Gen
    #     println("bus_ID=", gen.gen_ID, ", gen_i=", gen.gen_i, ", gbus=",gen.gbus, ", Qmax=",gen.Qmax, ", Qmin=",gen.Qmin,", Pmax=",gen.Pmax, ", Pmin=",gen.Pmin, ", cF2=",gen.cF2, ", cF1=",gen.cF1, ", cF0=",gen.cF0)
    # end

    # for line in pd.Line
    #     println("line_ID=", line.line_ID, ", line_i=", line.line_i, ", fbus=",line.fbus, ", tbus=",line.tbus,
    #     ", r=",line.r,", x=",line.x, ", b=",line.bc, ", rateA=",line.rateA, ", angmin=",line.angmin, ", angmax=",line.angmax
    #     , ", hi_bus=",line.hi_bus, ", gmd_br_hi=",line.gmd_br_hi, ", gmd_br_lo=",line.gmd_br_lo, ", gmd_k=",line.gmd_k, ", gmd_br_series=",line.gmd_br_series
    #     , ", gmd_br_common=",line.gmd_br_common, ", type=",line.type, ", config=",line.config
    #     )
    # end

    # for gmdbus in pd.GMDBus
    #     println("GMDbus_ID=", gmdbus.GMDbus_ID, ", GMDbus_i=",gmdbus.GMDbus_i, ", parent_bus=",gmdbus.parent_bus,  ", g_gnd=",gmdbus.g_gnd )
    # end

    # for gmdline in pd.GMDLine
    #     println("GMDline_ID=", gmdline.GMDline_ID, "GMDline_i=", gmdline.GMDline_i, "fbusd=", gmdline.fbusd, "tbusd=", gmdline.tbusd,
    #     "parent_branch=", gmdline.parent_branch, "br_r=", gmdline.br_r, "dist_E=", gmdline.dist_E, "dist_N=", gmdline.dist_N
    #     )
    # end

    ############################################ POST CALCULATIONS
    for line in pd.Line
        line.g = (line.r) / (line.r^2 + line.x^2)
        line.b = (-line.x) / (line.r^2 + line.x^2)
    end

    for bus in pd.Bus
        bus.wmax = bus.Vmax^2
        bus.wmin = bus.Vmin^2
    end

    pd.turn_ratio = Dict()
    pd.Immax = 0
    for line in pd.Line
        id_f = pd.Bus_index2id[line.fbus]
        id_t = pd.Bus_index2id[line.tbus]

        vmax_f = pd.Bus[id_f].Vmax
        vmax_t = pd.Bus[id_t].Vmax
        vmin_f = pd.Bus[id_f].Vmin
        vmin_t = pd.Bus[id_t].Vmin
        baseKV_f = pd.Bus[id_f].baseKV
        baseKV_t = pd.Bus[id_t].baseKV

        pd.turn_ratio[line.line_i] = baseKV_f / baseKV_t
        line.wcmax = vmax_f * vmax_t
        line.wcmin = 0.0
        line.wsmax = vmax_f * vmax_t * sin(line.angmax)
        line.wsmin = vmax_f * vmax_t * sin(line.angmin)

        # println("wcmax=", line.wcmax, ", wcmin=", line.wcmin, ", wsmax=", line.wsmax, ", wsmin=", line.wsmin )

        tmp_rateA = line.rateA * 100 * 1e6 ## VA
        temp_voltage_f = vmin_f * baseKV_f * 1e3 ## V
        temp_voltage_t = vmin_t * baseKV_t * 1e3 ## V
        # println( "tmp_rateA=", tmp_rateA, ", temp_voltage_f=", temp_voltage_f, ", temp_voltage_t=", temp_voltage_t)

        tmp = 2 * (tmp_rateA / min(temp_voltage_f, temp_voltage_t))
        if tmp > pd.Immax
            pd.Immax = tmp
        end
    end



end