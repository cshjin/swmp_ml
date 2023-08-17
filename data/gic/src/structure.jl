mutable struct BusData
    bus_ID::Int; bus_i::Int;  pd::Float64;  qd::Float64; gs::Float64; bs::Float64;  baseKV::Float64; Vmax::Float64; Vmin::Float64; lat::Float64; lon::Float64;
    wmax::Float64; wmin::Float64;
    BusData() = new()
end

mutable struct GenData
    gen_ID::Int; gen_i::Int;  gbus::Int;  Qmax::Float64; Qmin::Float64; Pmax::Float64; Pmin::Float64; cF2::Float64; cF1::Float64; cF0::Float64;

    GenData() = new()
end

mutable struct LineData
    line_ID::Int; line_i::Int;  fbus::Int;  tbus::Int;  r::Float64; x::Float64; bc::Float64; rateA::Float64; angmin::Float64; angmax::Float64;
    hi_bus::Int;  gmd_br_hi::Int;  gmd_br_lo::Int; gmd_k::Float64;  gmd_br_series::Int; gmd_br_common::Int;  type::String;  config::String;
    g::Float64; b::Float64; wcmax::Float64; wcmin::Float64; wsmax::Float64; wsmin::Float64;
    LineData() = new()
end

mutable struct GMDBusData
    GMDbus_ID::Int; GMDbus_i::Int;  parent_bus::Int; g_gnd::Float64;
    GMDBusData() = new()
end

mutable struct GMDLineData
    GMDline_ID::Int;  GMDline_i::Int;  fbusd::Int;  tbusd::Int;  parent_branch::Int;  br_r::Float64; dist_E::Float64; dist_N::Float64;
    GMDLineData() = new()
end

mutable struct PowerData
    Bus::Vector{}; Line::Vector{}; Gen::Vector{}; GMDBus::Vector{}; GMDLine::Vector{};
    Bus_index2id::Dict{};
    turn_ratio::Dict{}; Immax::Float64; scaling::Float64; baseMVA::Float64;
    numax::Float64; mu_E::Float64; mu_N::Float64;
    zg::Dict{}; za::Dict{}; dqloss::Dict{}; lpp::Dict{}; lpm::Dict{}; lqp::Dict{}; lqm::Dict{};
    penalty1::Float64; penalty2::Float64; vdmax::Float64; Scenario::Int64;
    xp_E::Dict{}; xp_N::Dict{};
    Setting_SOC::Int64; Setting_AC::Int64;
    Magnitude_Train::Vector{};Magnitude_Test::Vector{};
    Angle_Train::Vector{};Angle_Test::Vector{};
    tot_num_blockers::Int64; z::Dict{}
    PowerData() = new()
end


mutable struct Variables
    fp::Dict{}; fq::Dict{}; v::Dict{}; lpp::Dict{}; lpm::Dict{}; lqp::Dict{}; lqm::Dict{};
    w::Dict{}; theta::Dict{}; dqloss::Dict{}; p::Dict{}; q::Dict{}; wc::Dict{}; ws::Dict{};
    s::Dict{}; Ieff::Dict{}; vd::Dict{}; Id::Dict{}; ud::Dict{};
    z::Dict{}; u_mc::Dict{}; v_mc::Dict{};
    Ieff_plus::Dict{}; Ieff_minus::Dict{}
    vr::Dict{}; vi::Dict{}; wrr::Dict{}; wri::Dict{}; wii::Dict{};
    Variables() = new()
end
    
        