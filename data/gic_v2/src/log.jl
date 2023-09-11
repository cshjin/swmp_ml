function get_new_filename(prefix::AbstractString, ext::AbstractString)
  outfile = prefix
  if isfile(outfile * ext)
    num = 1
    numcopy = @sprintf("_%d", num)
    while isfile(outfile * numcopy * ext)
      num += 1
      numcopy = @sprintf("_%d", num)
    end
    outfile = outfile * numcopy
  end
  return outfile
end

function write_summary(args, m, var, pd)

  dir = args["output_dir_name"]
  if isdir(dir) == false
    mkpath(dir)
  end
  path = dir * args["output_file_name"]
  outfile = get_new_filename(path, ".txt")
  res_io = open(outfile * "_summary.txt", "w")

  @printf(res_io, "--results-- \n")
  @printf(res_io, "termination_status: % 20s \n", termination_status(m))
  @printf(res_io, "time: % 20s \n", args["elapsed_time"])
  if args["optimizer"] == "scip"
    @printf(res_io, "gap[percent]: % 20s \n", relative_gap(m) * 100)
    @printf(res_io, "LB: % 20s \n", objective_bound(m))
  end
  z = Dict()
  for gmdbus in pd.GMDBus
    i = gmdbus.GMDbus_i
    if gmdbus.g_gnd > 0.0 ## positive for substations
      if isempty(pd.z) == true
        z[i] = value(var.z[i])
      else
        z[i] = pd.z[i]
      end
    end
  end
  total_z = 0
  activate_z = []
  for gmdbus in pd.GMDBus
    i = gmdbus.GMDbus_i
    if gmdbus.g_gnd > 0.0 ## positive for substations
      total_z += 1
      if value(z[i]) > 0.95
        push!(activate_z, i)
      end
    end
  end
  gen_cost = 0.0
  for gen in pd.Gen
    k = gen.gen_i
    gen_cost = gen_cost + gen.cF1 * value(var.fp[k]) + gen.cF2 * value(var.fp[k])^2
  end
  shed_cost = 0.0
  for bus in pd.Bus
    i = bus.bus_i
    shed_cost = shed_cost + pd.penalty1 * (value(var.lpp[i]) + value(var.lpm[i]) + value(var.lqp[i]) + value(var.lqm[i]))
  end
  # block_cost = 0.0
  # if args["optimizer"] == "stochastic" || args["optimizer"] == "ipopt" 
  #     for gmdbus in pd.GMDBus
  #         i = gmdbus.GMDbus_i 
  #         if gmdbus.g_gnd > 0.0 ## positive for substations
  #             block_cost += pd.penalty2 * pd.z[i]
  #         end                                   
  #     end             
  # elseif args["optimizer"] == "trust" || args["optimizer"] == "admm"
  #     for gmdbus in pd.GMDBus
  #         i = gmdbus.GMDbus_i 
  #         if gmdbus.g_gnd > 0.0 ## positive for substations
  #             block_cost += pd.penalty2 * pd.z[i]
  #         end                                   
  #     end     

  # else
  #     for gmdbus in pd.GMDBus
  #         i = gmdbus.GMDbus_i 
  #         if gmdbus.g_gnd > 0.0 ## positive for substations
  #             block_cost += pd.penalty2 * value(var.z[i])
  #         end                                   
  #     end     
  # end


  @printf(res_io, "blockers: %20s \n", activate_z)
  @printf(res_io, "#blockers/total: %12s / %12s  \n", size(activate_z)[1], total_z)
  @printf(res_io, "objective_value: %20s \n", JuMP.objective_value(m))
  @printf(res_io, "gen_cost: %20s \n", gen_cost)
  @printf(res_io, "shed_cost: %20s \n", shed_cost)
  # @printf(res_io, "block_cost: %20s \n", block_cost)   


  flush(res_io)
  close(res_io)

end

function write_gen(args, var, pd)

  dir = args["output_dir_name"]
  path = dir * args["output_file_name"]
  outfile = get_new_filename(path, ".txt")
  res_io = open(outfile * "_gen.txt", "w")

  @printf(res_io,
    "%10s  %10s  %10s  %10s  %10s \n",
    "gen_id",
    "cost_1",
    "cost_2",
    "gen_p",
    "gen_q",
  )

  for gen in pd.Gen
    k = gen.gen_i
    @printf(res_io,
      "%10d  %10.4e  %10.4e  %10.4e  %10.4e \n",
      k,
      gen.cF1,
      gen.cF2,
      value(var.fp[k]),
      value(var.fq[k]),
    )
    flush(res_io)
  end

  close(res_io)
end

function write_bus(args, var, pd)

  dir = args["output_dir_name"]
  path = dir * args["output_file_name"]
  outfile = get_new_filename(path, ".txt")
  res_io = open(outfile * "_bus.txt", "w")

  @printf(res_io,
    "%10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s \n",
    "bus_id",
    "shed_cost",
    "bus_pd",
    "shed_lpp",
    "shed_lpm",
    "shed_lqp",
    "shed_lqm",
    "dqloss",
    "volt_mag_2"
  )

  for bus in pd.Bus
    i = bus.bus_i
    @printf(res_io,
      "%10d  %10.4e   %10.4e  %10.4e  %10.4e  %10.4e  %10.4e  %10.4e  %10.4e \n",
      i,
      pd.penalty1,
      bus.pd,
      value(var.lpp[i]),
      value(var.lpm[i]),
      value(var.lqp[i]),
      value(var.lqm[i]),
      value(var.dqloss[i]),
      value(var.w[i]),
    )
    flush(res_io)
  end

  close(res_io)
end

function write_line(args, var, pd)

  dir = args["output_dir_name"]
  path = dir * args["output_file_name"]
  outfile = get_new_filename(path, ".txt")
  res_io = open(outfile * "_line.txt", "w")

  @printf(res_io,
    "%10s  %10s  %10s  %10s  %10s  %10s  %10s %10s  \n",
    "line_id",
    "fr_bus",
    "to_bus",
    "p_fr",
    "p_to",
    "q_fr",
    "q_to",
    "Ieff"
  )

  Ieff = Dict()
  for line in pd.Line
    e = line.line_i
    i = line.fbus
    j = line.tbus


    if line.config == "gwye-delta" || line.config == "gwye-gwye-auto" || line.config == "gwye-gwye"
      Ieff[e] = value(var.Ieff[e])
    else
      Ieff[e] = -1.0
    end


    @printf(res_io,
      "%10d %10d %10d  %10.4e  %10.4e  %10.4e  %10.4e  %10.4e  \n",
      e,
      i,
      j,
      value(var.p[e, i]),
      value(var.p[e, j]),
      value(var.q[e, i]),
      value(var.q[e, j]),
      Ieff[e]
    )
    flush(res_io)
  end

  close(res_io)
end

function write_bus_dc(args, var, pd)

  dir = args["output_dir_name"]
  if isdir(dir) == false
    mkpath(dir)
  end

  path = dir * args["output_file_name"]
  outfile = get_new_filename(path, ".txt")
  res_io = open(outfile * "_bus_dc.txt", "w")

  @printf(res_io,
    "%10s  %10s  %10s \n",
    "bus_dc_id",
    "gic_blocker",
    "volt_mag",
  )

  z = Dict()
  for gmdbus in pd.GMDBus
    i = gmdbus.GMDbus_i
    if gmdbus.g_gnd > 0.0 ## positive for substations
      if isempty(pd.z) == true
        z[i] = value(var.z[i])
      else
        z[i] = pd.z[i]
      end
    else
      z[i] = -1.0
    end

    @printf(res_io,
      "%10d  %10.4e  %10.4e \n",
      i,
      z[i],
      value(var.vd[i]),
    )
    flush(res_io)
  end

  close(res_io)
end

function write_line_dc(args, var, pd)

  dir = args["output_dir_name"]
  path = dir * args["output_file_name"]
  outfile = get_new_filename(path, ".txt")
  res_io = open(outfile * "_line_dc.txt", "w")

  @printf(res_io,
    "%10s  %10s  \n",
    "line_dc_id",
    "gic_current",
  )

  for gmdline in pd.GMDLine
    e = gmdline.GMDline_i

    @printf(res_io,
      "%10d  %10.4e \n",
      e,
      value(var.Id[e]),
    )
    flush(res_io)
  end

  close(res_io)
end

function write_json(args, var, pd)
  #= NOTE: just a simple extract of pd/qd perturbations with label 'gic_blocker' to json format =#
  dir = args["output_dir_name"]
  if isdir(dir) == false
    mkpath(dir)
  end

  dict_gmdbus = Dict()
  for gmdbus in pd.GMDBus
    i = gmdbus.GMDbus_i
    gmdbus_i = Dict()
    if gmdbus.g_gnd > 0.0
      if isempty(pd.z) == true
        gmdbus_i["gic_blocker"] = value(var.z[i])
      else
        gmdbus_i["gic_blocker"] = pd.z[i]
      end
    else
      gmdbus_i["gic_blocker"] = -1.0
    end
    gmdbus_i["volt_mag"] = value(var.vd[i])

    dict_gmdbus[i] = gmdbus_i
  end

  dict_bus = Dict()
  for bus in pd.Bus
    i = bus.bus_i
    bus_i = Dict()
    bus_i["pd"] = bus.pd
    bus_i["qd"] = bus.qd
    dict_bus[i] = bus_i
  end

  open(dir * args["output_file_name"] * ".json", "w") do f
    JSON.print(f, Dict("bus" => dict_bus, "gmd_bus" => dict_gmdbus), 4)
  end
end