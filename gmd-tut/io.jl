using Printf, DataFrames

function print_dict(x; drop=["index","bus_i"])
    drop_set = Set(drop)

    for (k,y) in x
        if k in drop_set
            continue
        end

        println("$k: $y")
    end
end

function print_bus(case, bid)
    bus = case["bus"]["$bid"]

    # print the loads that connect to this bus
    println("Bus $bid")
    print_dict(bus)
    println()

    bus_loads = [x for x in values(case["load"]) if x["load_bus"] == bid]

    if length(bus_loads) > 0
        println("Loads:")
        @printf "%5s  %8s, %8s, %8s\n" "" "index" "pd" "qd"

        for (i,x) in enumerate(bus_loads)
            @printf "%5d: %8d, %8.3f, %8.3f\n" i x["index"] x["pd"] x["qd"]
        end
    else
        println("No attached loads")
    end

    # print the generators that connect 
    bus_gens = [x for x in values(case["gen"]) if x["gen_bus"] == bid]

    if length(bus_gens) > 0
        println()
        println("Generators:")
        @printf "%5s  %8s, %8s, %8s, %8s, %8s, %8s, %8s\n" "" "index" "pg" "qg" "pmin" "pmax" "qmin" "qmax"
    

        for (i,x) in enumerate(bus_gens)
            @printf "%5d: %8d, %8.3f, %8.3f, %8.3f, %8.3f, %8.3f, %8.3f\n" i x["index"] x["pg"] x["qg"] x["pmin"] x["pmax"] x["qmin"] x["qmax"]
        end
    else
        println("No attached generators")
    end
        
    # print the branches that connect 
    bus_branches = [x for x in values(case["branch"]) if (x["f_bus"] == bid || x["t_bus"] == bid)]

    if length(bus_branches) > 0
        println()
        println("Branches")
        # f_bus, t_bus,  br_r,  br_x, shift, rate_a, rate_b, rate_c
        @printf "%5s  %8s, %8s, %8s %8s, %8s, %8s, %8s\n" "" "f_bus" "t_bus" "is_xf" "br_r" "br_x" "shift" "rate_a"


        for (i,x) in enumerate(bus_branches)
            @printf "%5d: %8d, %8d, %8d %8.3f, %8.3f, %8.3f, %8.3f\n" i x["f_bus"] x["t_bus"] x["transformer"] x["br_r"] x["br_x"] x["shift"] x["rate_a"]
        end
    end

end

# Note: use names() to print the column names

function to_df(case, table_name, result=nothing)
    df = DataFrame()
    table = case[table_name]
    ids = sort(map(x->x["index"], values(table)))
    df[!, :index] = ids

    for k in keys(first(values(table)))
        col = []

        for i in ids
            row = table["$i"]

            if k in keys(row)
                push!(col, row[k])
            else
                push!(col, nothing)
            end
        end

        df[!, Symbol(k)] = col 
    end 
    
    if result !== nothing
        soln_table = result["solution"][table_name]

        for k in keys(first(values(soln_table)))
            col = []

            for i in ids
                if "$i" in keys(soln_table)
                    row = soln_table["$i"]
                    
                    if k in keys(row)
                        push!(col, row[k])
                    else
                        push!(col, nothing)
                    end
                else 
                    push!(col, nothing)
                end
            end

            # df[!, Symbol(k)] = zeros(length(table))
            # df[df[!,"br_status"].==true, Symbol(k)] = col 
            df[!, Symbol("s_$k")] = col 
        end 
    end
    
    return df
end

function to_csv(file, case, table_name, result=nothing)
    f = file

    if typeof(file) == String
        f = open(file, "w")
    end

    table = case[table_name]
    cols = []
    fields = []

    rec = first(values(table))

    for c in keys(rec)
        for rec2 in values(table)
            if !(c in keys(rec2))
                rec2[c] = nothing
            end
        end

        if typeof(rec[c]) <: Array
            for j in 1:length(rec[c])
                push!(cols, "$c$j")
            end
        elseif typeof(rec[c]) <: Dict
            continue
        else
            push!(cols, c)
        end

        push!(fields, c)
    end

    i = 1
    for c in cols
        if i >= 2 
            print(f, ",")
        end

        print(f, "$c")
        i += 1
    end      

    soln_table = Dict()
    soln_cols = []
    soln_fields = []

    if result !== nothing
       
        soln = result
        if "solution" in keys(result) 
            soln = result["solution"]
        end

        soln_table = Dict()

        if table_name in keys(soln)
            soln_table = soln[table_name]    

            soln_rec = first(values(soln_table))

            for c in keys(soln_rec)
                if typeof(soln_rec[c]) <: Array 
                    for j in 1:length(soln_rec[c])
                        push!(soln_cols, "s_$c$j")
                    end
                elseif typeof(rec[c]) <: Dict
                    continue                
                else
                    push!(soln_cols, c)
                end

                push!(soln_fields, c)
            end
        end
    end    
    
    for c in soln_cols
        if i >= 2 
            print(f, ",")
        end

        print(f, "$c")
        i += 1
    end      

    println(f, "")
    # println("$(length(fields)) fields discovered")
    # println("$(length(cols)) columns discovered")


    for r in keys(table) 
        row = table[r]
        if table_name == "bus" && "name" in keys(row) && startswith(row["name"],"_virtual")
            continue
        end

        if table_name == "branch" && "source_id" in keys(row) && startswith(row["source_id"],"_virtual")
            continue
        end

        i = 1
        for fi in fields
            if fi in keys(row)
                val = row[fi]

                if typeof(val) <: Array 
                    for x in val
                        if i >= 2 
                            print(f, ",")
                        end

                        print(f, x)
                        i += 1
                    end
                else
                    if i >= 2 
                        print(f, ",")
                    end

                    print(f, val)
                    i += 1
                end
            end
        end 

        if r in keys(soln_table) 
            row = soln_table[r]

            for fi in soln_fields
                if fi in keys(row)
                    val = row[fi]
    
                    if typeof(val) <: Array 
                        for x in val
                            if i >= 2 
                                print(f, ",")
                            end
    
                            print(f, x)
                            i += 1
                        end
                    else
                        if i >= 2 
                            print(f, ",")
                        end
                        
                        print(f, val)
                        i += 1
                    end
                end
            end 
        end   
        println(f, "")
    end

    
    if typeof(file) == String
        close(f)
    end
end

function print_row(f, cols, fields, indexes, row)
    for c in cols
        fi = fields[c]
        i = indexes[c]
        print(f, ",")

        if fi in keys(row)
            val = row[fi]

            if i == -1
                print(f, val)
            elseif  i > 0 && length(val) >= i
                print(f, val[i])
            end
        end
    end 
end

function engr_to_csv(file, case, table_name, soln=nothing)
    f = file

    if typeof(file) == String
        f = open(file, "w")
    end

    
    if "solution" in keys(soln)
        soln = soln["solution"]
    end

    if table_name in keys(soln)
        soln = soln[table_name]
    end

    table = case[table_name]

    if soln !== nothing 
        for r in keys(table)
            # println("Updating $r")

            if r in keys(soln)
                for c in keys(soln[r])
                    table[r]["s_$c"] = soln[r][c]
                end
            end 
        end
    end

    rec = Dict()

    for obj in values(table)
        for c in keys(obj)
            if !(c in keys(rec))
                rec[c] = obj[c]
            elseif typeof(obj[c]) <: Array && length(obj[c]) > length(rec[c])
                rec[c] = obj[c]
            end
        end
    end

    cols = []
    fields = Dict()
    indexes = Dict()

    for c in keys(rec)
        if typeof(rec[c]) <: Array
            for j in 1:length(rec[c])
                push!(cols, "$c$j")
                fields["$c$j"] = c
                indexes["$c$j"] = j
            end
        elseif typeof(rec[c]) <: Dict
            continue
        else
            fields[c] = c
            indexes[c] = -1
        end
    end

    print(f,"name")
    for c in cols
        print(f, ",")
         print(f, "$c")
    end      


    println(f, "")
    # println("$(length(fields)) fields discovered")
    # println("$(length(cols)) columns discovered")


    for (r,row) in table
        print(f, "$r")
        print_row(f, cols, fields, indexes, row)
        println(f, "")
    end

    
    if typeof(file) == String
        close(f)
    end
end

function write_mods(onet, data, modsfile="mods.json")
    omods = Dict()
    omods["branch"] = Dict()
    omods["gen"] = Dict()
    omods["load"] = Dict()
    omods["bus"] = Dict()
    omods["dcline"] = Dict()
    
    s = data["solution"]
    
    for (k,b) in onet["bus"]
        if !("bus" in keys(s)) || !(k in keys(s["bus"])) || b["bus_type"] == 4 || s["bus"][k]["status"] <= 0.1
            sid = ("bus", b["index"])
            omods["bus"][k] = Dict("bus_type" => 4, "source_id" => sid) 
        end
    end
    
    for (k,b) in onet["branch"]
        if !("branch" in keys(s)) || !(k in keys(s["branch"])) || b["br_status"] == 0
            sid = b["source_id"]
            
            if "ckt" in keys(b)
                sid = ("branch", b["f_bus"], b["t_bus"], b["ckt"])
            end

            omods["branch"][k] = Dict("br_status" => 0, "source_id" => sid) 
        end
    end
    
    for (k,g) in onet["gen"]
        if !("gen" in keys(s)) || !(k in keys(s["gen"])) || s["gen"][k]["gen_status"] <= 0.1
            # sid = ("gen", g["gen_bus"], g["gen_id"])
            sid= g["source_id"]
            omods["gen"][k] = Dict("gen_status" => 0, "source_id" => sid) 
        end
    end
    
    for (k,d) in onet["load"]
        if !("load" in keys(s)) || !(k in keys(s["load"])) 
            if "status" in keys(s["load"]) && s["load"][k]["status"] <= 0.1
                sid = ("load", d["load_bus"], "1 ")
                omods["load"][k] = Dict("status" => 0, "source_id" => sid) 
            end
        end
    end
    
    for (k,d) in onet["dcline"]
        if !("dcline" in keys(s)) || !(k in keys(s["dcline"])) || s["dcline"][k]["status"] <= 0.1
            # sid = ("dcline", d["dcline_bus"], "1 ")
            sid = d["source_id"]
            omods["dcline"][k] = Dict("status" => 0, "source_id" => sid) 
        end
    end

    f = open(modsfile, "w")
    JSON.print(f, omods)
    close(f)
end
