%% MATPOWER Case Format : Version 2
function mpc = case24_ieee_rts_0
mpc.version = '2';


%%-----  Power Flow Data  -----%%

%% system MVA base
mpc.baseMVA = 100;


%% bus data
%    bus_i    type    Pd    Qd    Gs    Bs    area    Vm    Va    baseKV    zone    Vmax    Vmin
mpc.bus = [
	1	3	108	22	0	0	1	1.100000	0.000000	138	1	1.05	0.95
	2	1	97	20	0	0	1	1.100000	0.000000	138	1	1.05	0.95
	3	1	180	37	0	0	1	1.100000	0.000000	138	1	1.05	0.95
	4	1	74	15	0	0	1	1.100000	0.000000	138	1	1.05	0.95
	5	1	71	14	0	0	1	1.100000	0.000000	138	1	1.05	0.95
	6	1	136	28	0	-100	1	1.100000	0.000000	138	1	1.05	0.95
	7	1	125	25	0	0	1	1.100000	0.000000	138	1	1.05	0.95
	8	1	171	35	0	0	1	1.100000	0.000000	138	1	1.05	0.95
	9	1	175	36	0	0	1	1.100000	0.000000	138	1	1.05	0.95
	10	1	195	40	0	0	1	1.100000	0.000000	138	1	1.05	0.95
	11	1	0	0	0	0	1	1.100000	0.000000	230	1	1.05	0.95
	12	1	0	0	0	0	1	1.100000	0.000000	230	1	1.05	0.95
	13	1	265	54	0	0	1	1.100000	0.000000	230	1	1.05	0.95
	14	1	194	39	0	0	1	1.100000	0.000000	230	1	1.05	0.95
	15	1	317	64	0	0	1	1.100000	0.000000	230	1	1.05	0.95
	16	1	100	20	0	0	1	1.100000	0.000000	230	1	1.05	0.95
	17	1	0	0	0	0	1	1.100000	0.000000	230	1	1.05	0.95
	18	1	333	68	0	0	1	1.100000	0.000000	230	1	1.05	0.95
	19	1	181	37	0	0	1	1.100000	0.000000	230	1	1.05	0.95
	20	1	128	26	0	0	1	1.100000	0.000000	230	1	1.05	0.95
	21	1	0	0	0	0	1	1.100000	0.000000	230	1	1.05	0.95
	22	1	0	0	0	0	1	1.100000	0.000000	230	1	1.05	0.95
	23	1	0	0	0	0	1	1.100000	0.000000	230	1	1.05	0.95
	24	1	0	0	0	0	1	1.100000	0.000000	230	1	1.05	0.95
	25	2	0	0	0	0	1	1.100000	0.000000	138	1	1.05	0.95
	26	2	0	0	0	0	1	1.100000	0.000000	138	1	1.05	0.95
	27	2	0	0	0	0	1	1.100000	0.000000	138	1	1.05	0.95
	28	2	0	0	0	0	1	1.100000	0.000000	138	1	1.05	0.95
	29	2	0	0	0	0	1	1.100000	0.000000	138	1	1.05	0.95
	30	2	0	0	0	0	1	1.100000	0.000000	138	1	1.05	0.95
	31	2	0	0	0	0	1	1.100000	0.000000	138	1	1.05	0.95
	32	2	0	0	0	0	1	1.100000	0.000000	138	1	1.05	0.95
	33	2	0	0	0	0	1	1.100000	0.000000	138	1	1.05	0.95
	34	2	0	0	0	0	1	1.100000	0.000000	138	1	1.05	0.95
	35	2	0	0	0	0	1	1.100000	0.000000	138	1	1.05	0.95
	36	2	0	0	0	0	1	1.100000	0.000000	230	1	1.05	0.95
	37	2	0	0	0	0	1	1.100000	0.000000	230	1	1.05	0.95
	38	2	0	0	0	0	1	1.100000	0.000000	230	1	1.05	0.95
	39	2	0	0	0	0	1	1.100000	0.000000	230	1	1.05	0.95
	40	2	0	0	0	0	1	1.100000	0.000000	230	1	1.05	0.95
	41	2	0	0	0	0	1	1.100000	0.000000	230	1	1.05	0.95
	42	2	0	0	0	0	1	1.100000	0.000000	230	1	1.05	0.95
	43	2	0	0	0	0	1	1.100000	0.000000	230	1	1.05	0.95
	44	2	0	0	0	0	1	1.100000	0.000000	230	1	1.05	0.95
	45	2	0	0	0	0	1	1.100000	0.000000	230	1	1.05	0.95
	46	2	0	0	0	0	1	1.100000	0.000000	230	1	1.05	0.95
	47	2	0	0	0	0	1	1.100000	0.000000	230	1	1.05	0.95
	48	2	0	0	0	0	1	1.100000	0.000000	230	1	1.05	0.95
	49	2	0	0	0	0	1	1.100000	0.000000	230	1	1.05	0.95
	50	2	0	0	0	0	1	1.100000	0.000000	230	1	1.05	0.95
	51	2	0	0	0	0	1	1.100000	0.000000	230	1	1.05	0.95
	52	2	0	0	0	0	1	1.100000	0.000000	230	1	1.05	0.95
	53	2	0	0	0	0	1	1.100000	0.000000	230	1	1.05	0.95
	54	2	0	0	0	0	1	1.100000	0.000000	230	1	1.05	0.95
	55	2	0	0	0	0	1	1.100000	0.000000	230	1	1.05	0.95
	56	2	0	0	0	0	1	1.100000	0.000000	230	1	1.05	0.95
	57	2	0	0	0	0	1	1.100000	0.000000	230	1	1.05	0.95
];


%% generator data
%    bus    Pg    Qg    Qmax    Qmin    Vg    mBase    status    Pmax    Pmin    Pc1    Pc2    Qc1min    Qc1max    Qc2min    Qc2max    ramp_agc    ramp_10    ramp_30    ramp_q    apf
mpc.gen = [
	25	10.0	0.0	10.0	0.0	1.1	100	1	20.0	16.0	0	0	0	0	0	0	0	0	0	-1000	0
	34	80.0	0.0	60.0	0.0	1.1	100	1	100.0	25.0	0	0	0	0	0	0	0	0	0	-1000	0
	35	80.0	0.0	60.0	0.0	1.1	100	1	100.0	25.0	0	0	0	0	0	0	0	0	0	-1000	0
	36	95.1	0.0	80.0	0.0	1.1	100	1	197.0	69.0	0	0	0	0	0	0	0	0	0	-1000	0
	37	95.1	0.0	80.0	0.0	1.1	100	1	197.0	69.0	0	0	0	0	0	0	0	0	0	-1000	0
	38	95.1	0.0	80.0	0.0	1.1	100	1	197.0	69.0	0	0	0	0	0	0	0	0	0	-1000	0
	39	0.0	35.3	200.0	-50.0	1.1	100	1	0.0	0.0	0	0	0	0	0	0	0	0	0	-1000	0
	40	12.0	0.0	6.0	0.0	1.1	100	1	12.0	2.4	0	0	0	0	0	0	0	0	0	-1000	0
	41	12.0	0.0	6.0	0.0	1.1	100	1	12.0	2.4	0	0	0	0	0	0	0	0	0	-1000	0
	42	12.0	0.0	6.0	0.0	1.1	100	1	12.0	2.4	0	0	0	0	0	0	0	0	0	-1000	0
	43	12.0	0.0	6.0	0.0	1.1	100	1	12.0	2.4	0	0	0	0	0	0	0	0	0	-1000	0
	26	10.0	0.0	10.0	0.0	1.1	100	1	20.0	16.0	0	0	0	0	0	0	0	0	0	-1000	0
	44	12.0	0.0	6.0	0.0	1.1	100	1	12.0	2.4	0	0	0	0	0	0	0	0	0	-1000	0
	45	155.0	0.0	80.0	-50.0	1.1	100	1	155.0	54.29999999999999	0	0	0	0	0	0	0	0	0	-1000	0
	46	155.0	0.0	80.0	-50.0	1.1	100	1	155.0	54.29999999999999	0	0	0	0	0	0	0	0	0	-1000	0
	47	400.0	0.0	200.0	-50.0	1.1	100	1	400.0	100.0	0	0	0	0	0	0	0	0	0	-1000	0
	48	400.0	0.0	200.0	-50.0	1.1	100	1	400.0	100.0	0	0	0	0	0	0	0	0	0	-1000	0
	49	50.0	0.0	16.0	-10.0	1.1	100	1	50.0	10.0	0	0	0	0	0	0	0	0	0	-1000	0
	50	50.0	0.0	16.0	-10.0	1.1	100	1	50.0	10.0	0	0	0	0	0	0	0	0	0	-1000	0
	51	50.0	0.0	16.0	-10.0	1.1	100	1	50.0	10.0	0	0	0	0	0	0	0	0	0	-1000	0
	52	50.0	0.0	16.0	-10.0	1.1	100	1	50.0	10.0	0	0	0	0	0	0	0	0	0	-1000	0
	53	50.0	0.0	16.0	-10.0	1.1	100	1	50.0	10.0	0	0	0	0	0	0	0	0	0	-1000	0
	27	76.0	0.0	30.0	-25.0	1.1	100	1	76.0	15.2	0	0	0	0	0	0	0	0	0	-1000	0
	54	50.0	0.0	16.0	-10.0	1.1	100	1	50.0	10.0	0	0	0	0	0	0	0	0	0	-1000	0
	55	155.0	0.0	80.0	-50.0	1.1	100	1	155.0	54.29999999999999	0	0	0	0	0	0	0	0	0	-1000	0
	56	155.0	0.0	80.0	-50.0	1.1	100	1	155.0	54.29999999999999	0	0	0	0	0	0	0	0	0	-1000	0
	57	350.0	0.0	150.0	-25.0	1.1	100	1	350.0	140.0	0	0	0	0	0	0	0	0	0	-1000	0
	28	76.0	0.0	30.0	-25.0	1.1	100	1	76.0	15.2	0	0	0	0	0	0	0	0	0	-1000	0
	29	10.0	0.0	10.0	0.0	1.1	100	1	20.0	16.0	0	0	0	0	0	0	0	0	0	-1000	0
	30	10.0	0.0	10.0	0.0	1.1	100	1	20.0	16.0	0	0	0	0	0	0	0	0	0	-1000	0
	31	76.0	0.0	30.0	-25.0	1.1	100	1	76.0	15.2	0	0	0	0	0	0	0	0	0	-1000	0
	32	76.0	0.0	30.0	-25.0	1.1	100	1	76.0	15.2	0	0	0	0	0	0	0	0	0	-1000	0
	33	80.0	0.0	60.0	0.0	1.1	100	1	100.0	25.0	0	0	0	0	0	0	0	0	0	-1000	0
];


%% branch data
%    fbus    tbus    r    x    b    rateA    rateB    rateC    ratio    angle    status    angmin    angmax
mpc.branch = [
	1	2	0.003452	0.018456	0.347267	175.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	6	10	0.017341	0.075477	1.971055	175.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	7	8	0.015942	0.061561	0.016557	175.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	8	9	0.04321	0.167071	0.044173	175.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	8	10	0.04321	0.167071	0.044173	400.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	11	13	0.006646	0.051858	0.091697	500.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	11	14	0.006327	0.048973	0.075025	500.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	1	3	0.052766	0.204107	0.059188	175.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	12	13	0.006646	0.051858	0.091697	500.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	12	23	0.013044	0.101614	0.192983	500.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	13	23	0.010616	0.082732	0.19008	500.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	14	16	0.005066	0.039416	0.08073	500.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	15	16	0.002062	0.016212	0.038844	500.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	15	21	0.006567	0.051075	0.098815	500.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	15	21	0.006567	0.051075	0.098815	500.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	15	24	0.007152	0.055401	0.102206	500.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	16	17	0.003441	0.027008	0.052264	500.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	16	19	0.003481	0.026805	0.041796	500.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	1	5	0.022572	0.087493	0.022117	175.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	17	18	0.001935	0.015479	0.028187	500.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	17	22	0.013471	0.10507	0.221684	500.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	18	21	0.003293	0.025845	0.054616	500.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	18	21	0.003293	0.025845	0.054616	500.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	19	20	0.005559	0.043161	0.076427	500.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	19	20	0.005559	0.043161	0.076427	500.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	20	23	0.002911	0.022455	0.043767	500.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	20	23	0.002911	0.022455	0.043767	500.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	21	22	0.009594	0.074769	0.129128	500.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	2	4	0.032962	0.127324	0.034132	175.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	2	6	0.044219	0.170826	0.058446	175.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	3	9	0.033342	0.128823	0.029745	175.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	4	9	0.026689	0.103271	0.028217	175.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	5	10	0.023179	0.089769	0.023509	175.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	3	24	0.0023	0.0839	0	400.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	2	29	0.0023	0.0839	0	23.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	2	30	0.0023	0.0839	0	23.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	2	31	0.0023	0.0839	0	82.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	2	32	0.0023	0.0839	0	82.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	7	33	0.0023	0.0839	0	117.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	7	34	0.0023	0.0839	0	117.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	7	35	0.0023	0.0839	0	117.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	13	36	0.0023	0.0839	0	213.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	13	37	0.0023	0.0839	0	213.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	13	38	0.0023	0.0839	0	213.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	9	11	0.0023	0.0839	0	400.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	14	39	0.0023	0.0839	0	200.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	15	40	0.0023	0.0839	0	14.000000000000002	0.0	0.0	1.0	0.0	1	-30.0	30.0
	15	41	0.0023	0.0839	0	14.000000000000002	0.0	0.0	1.0	0.0	1	-30.0	30.0
	15	42	0.0023	0.0839	0	14.000000000000002	0.0	0.0	1.0	0.0	1	-30.0	30.0
	15	43	0.0023	0.0839	0	14.000000000000002	0.0	0.0	1.0	0.0	1	-30.0	30.0
	15	44	0.0023	0.0839	0	14.000000000000002	0.0	0.0	1.0	0.0	1	-30.0	30.0
	15	45	0.0023	0.0839	0	175.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	16	46	0.0023	0.0839	0	175.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	18	47	0.0023	0.0839	0	448.00000000000006	0.0	0.0	1.0	0.0	1	-30.0	30.0
	21	48	0.0023	0.0839	0	448.00000000000006	0.0	0.0	1.0	0.0	1	-30.0	30.0
	9	12	0.0023	0.0839	0	400.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	22	49	0.0023	0.0839	0	53.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	22	50	0.0023	0.0839	0	53.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	22	51	0.0023	0.0839	0	53.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	22	52	0.0023	0.0839	0	53.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	22	53	0.0023	0.0839	0	53.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	22	54	0.0023	0.0839	0	53.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	23	55	0.0023	0.0839	0	175.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	23	56	0.0023	0.0839	0	175.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	23	57	0.0023	0.0839	0	381.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	10	11	0.0023	0.0839	0	400.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	10	12	0.0023	0.0839	0	400.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	1	25	0.0023	0.0839	0	23.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	1	26	0.0023	0.0839	0	23.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	1	27	0.0023	0.0839	0	82.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
	1	28	0.0023	0.0839	0	82.0	0.0	0.0	1.0	0.0	1	-30.0	30.0
];


%%-----  OPF Data  -----%%

%% generator cost data
%    1    startup    shutdown    n    x1    y1    ...    xn    yn
%    2    startup    shutdown    n    c(n-1)    ...    c0
mpc.gencost = [
	2	0	0	3	0.0	130.0	400.685
	2	0	0	3	0.05267199999999999	43.6615	781.521
	2	0	0	3	0.05267199999999999	43.6615	781.521
	2	0	0	3	0.00717	48.5804	832.758
	2	0	0	3	0.00717	48.5804	832.758
	2	0	0	3	0.00717	48.5804	832.758
	2	0	0	3	0.0	0.0	0
	2	0	0	3	0.328412	56.56399999999999	86.3852
	2	0	0	3	0.328412	56.56399999999999	86.3852
	2	0	0	3	0.328412	56.56399999999999	86.3852
	2	0	0	3	0.328412	56.56399999999999	86.3852
	2	0	0	3	0.0	130.0	400.685
	2	0	0	3	0.328412	56.56399999999999	86.3852
	2	0	0	3	0.008342	12.3883	382.239
	2	0	0	3	0.008342	12.3883	382.239
	2	0	0	3	0.000213	4.4231	395.375
	2	0	0	3	0.000213	4.4231	395.375
	2	0	0	3	0.0	0.001	0.001
	2	0	0	3	0.0	0.001	0.001
	2	0	0	3	0.0	0.001	0.001
	2	0	0	3	0.0	0.001	0.001
	2	0	0	3	0.0	0.001	0.001
	2	0	0	3	0.014141999999999998	16.0811	212.308
	2	0	0	3	0.0	0.001	0.001
	2	0	0	3	0.008342	12.3883	382.239
	2	0	0	3	0.008342	12.3883	382.239
	2	0	0	3	0.004895	11.8495	665.109
	2	0	0	3	0.014141999999999998	16.0811	212.308
	2	0	0	3	0.0	130.0	400.685
	2	0	0	3	0.0	130.0	400.685
	2	0	0	3	0.014141999999999998	16.0811	212.308
	2	0	0	3	0.014141999999999998	16.0811	212.308
	2	0	0	3	0.05267199999999999	43.6615	781.521
];


%%-----  GMD - Thermal Data  -----%%

%% gmd_bus data
%column_names% parent_index status g_gnd name
mpc.gmd_bus = {
	1	1   10	'dc_sub1'
	2	1   10	'dc_sub2'
	3	1   10	'dc_sub3'
	4	1   10	'dc_sub4'
	5	1   10	'dc_sub5'
	6	1   10	'dc_sub6'
	7	1   10	'dc_sub7'
	8	1   10	'dc_sub8'
	9	1   10	'dc_sub9'
	10	1   10	'dc_sub10'
	11	1   10	'dc_sub11'
	12	1   10	'dc_sub12'
	13	1   10	'dc_sub13'
	14	1   10	'dc_sub14'
	15	1   10	'dc_sub15'
	16	1   10	'dc_sub16'
	17	1   10	'dc_sub17'
	18	1   10	'dc_sub18'
	19	1   10	'dc_sub19'
	20	1   10	'dc_sub20'
	1	1   0	'dc_bus1'
	2	1   0	'dc_bus2'
	3	1   0	'dc_bus3'
	4	1   0	'dc_bus4'
	5	1   0	'dc_bus5'
	6	1   0	'dc_bus6'
	7	1   0	'dc_bus7'
	8	1   0	'dc_bus8'
	9	1   0	'dc_bus9'
	10	1   0	'dc_bus10'
	11	1   0	'dc_bus11'
	12	1   0	'dc_bus12'
	13	1   0	'dc_bus13'
	14	1   0	'dc_bus14'
	15	1   0	'dc_bus15'
	16	1   0	'dc_bus16'
	17	1   0	'dc_bus17'
	18	1   0	'dc_bus18'
	19	1   0	'dc_bus19'
	20	1   0	'dc_bus20'
	21	1   0	'dc_bus21'
	22	1   0	'dc_bus22'
	23	1   0	'dc_bus23'
	24	1   0	'dc_bus24'
	25	1   0	'dc_bus25'
	26	1   0	'dc_bus26'
	27	1   0	'dc_bus27'
	28	1   0	'dc_bus28'
	29	1   0	'dc_bus29'
	30	1   0	'dc_bus30'
	31	1   0	'dc_bus31'
	32	1   0	'dc_bus32'
	33	1   0	'dc_bus33'
	34	1   0	'dc_bus34'
	35	1   0	'dc_bus35'
	36	1   0	'dc_bus36'
	37	1   0	'dc_bus37'
	38	1   0	'dc_bus38'
	39	1   0	'dc_bus39'
	40	1   0	'dc_bus40'
	41	1   0	'dc_bus41'
	42	1   0	'dc_bus42'
	43	1   0	'dc_bus43'
	44	1   0	'dc_bus44'
	45	1   0	'dc_bus45'
	46	1   0	'dc_bus46'
	47	1   0	'dc_bus47'
	48	1   0	'dc_bus48'
	49	1   0	'dc_bus49'
	50	1   0	'dc_bus50'
	51	1   0	'dc_bus51'
	52	1   0	'dc_bus52'
	53	1   0	'dc_bus53'
	54	1   0	'dc_bus54'
	55	1   0	'dc_bus55'
	56	1   0	'dc_bus56'
	57	1   0	'dc_bus57'
};


%% gmd_branch data
%column_names% f_bus t_bus parent_index br_status br_r br_v len_km name
mpc.gmd_branch = {
	21	22	1	1	0.21913296	3.9345022195631e-15	6.4255297483363	'dc_br1'
	26	30	2	1	1.10080668	-48.304620679561	32.196878797541	'dc_br10'
	27	28	3	1	1.01199816	32.200031997548	25.875624035482	'dc_br11'
	28	29	4	1	2.7429708	563.52685952678	70.01706570387	'dc_br12'
	28	30	5	1	2.7429708	563.52685952678	70.01706570387	'dc_br13'
	31	33	6	1	1.1719113333333	209.32266903755	57.967077977355	'dc_br18'
	31	34	7	1	1.115661	499.16542022511	54.636314868195	'dc_br19'
	21	23	8	1	3.34958568	515.21480503094	85.62152789958	'dc_br2'
	32	33	9	1	1.1719113333333	209.32266903755	57.967077977355	'dc_br20'
	32	43	10	1	2.300092	1096.6497006498	113.29358943973	'dc_br21'
	33	43	11	1	1.8719546666667	887.32704739153	92.25438860412	'dc_br22'
	34	36	12	1	0.89330466666667	161.02946073616	44.111403726089	'dc_br23'
	35	36	13	1	0.36359933333333	193.23486336698	19.580115928517	'dc_br24'
	35	41	14	1	1.157981	547.51404073997	56.974421505833	'dc_br25'
	35	41	15	1	1.157981	547.51404073997	56.974421505833	'dc_br26'
	35	44	16	1	1.261136	-611.87167924016	61.759806095831	'dc_br27'
	36	37	17	1	0.606763	257.6556256995	30.19764739266	'dc_br28'
	36	39	18	1	0.61381633333333	161.03354139411	29.918911521577	'dc_br29'
	21	25	19	1	1.43287056	289.80321075598	36.656153520373	'dc_br3'
	37	38	20	1	0.341205	161.04007537334	17.285607726605	'dc_br30'
	37	42	21	1	2.3753863333333	611.97402968748	117.39278204147	'dc_br31'
	38	41	22	1	0.58066566666667	-64.416520456267	28.971296492166	'dc_br32'
	38	41	23	1	0.58066566666667	-64.416520456267	28.971296492166	'dc_br33'
	39	40	24	1	0.980237	257.66215968961	48.30175284337	'dc_br34'
	39	40	25	1	0.980237	257.66215968961	48.30175284337	'dc_br35'
	40	43	26	1	0.51330633333333	17.75914774518	25.155858586499	'dc_br36'
	40	43	27	1	0.51330633333333	17.75914774518	25.155858586499	'dc_br37'
	41	42	28	1	1.691742	515.35047663811	83.498896512283	'dc_br38'
	22	24	29	1	2.09242776	289.80321075598	53.433297995742	'dc_br4'
	22	26	30	1	2.80702212	708.43108378282	71.489606559925	'dc_br5'
	23	29	31	1	2.11655016	144.91166459486	54.119952610789	'dc_br6'
	24	29	32	1	1.69421772	370.3232609381	43.252121505118	'dc_br8'
	25	30	33	1	1.47140292	370.3232609381	37.578673113366	'dc_br9'
	23	44	34	1	0.02	0	0	'dc_xf1_series'
	23	3	34	1	0.013333333333333	0	0	'dc_xf1_common'
	22	2	35	1	0.033333333333333	0	0	'dc_xf10_hi'
	22	2	36	1	0.033333333333333	0	0	'dc_xf11_hi'
	22	2	37	1	0.033333333333333	0	0	'dc_xf12_hi'
	22	2	38	1	0.033333333333333	0	0	'dc_xf13_hi'
	27	7	39	1	0.033333333333333	0	0	'dc_xf14_hi'
	27	7	40	1	0.033333333333333	0	0	'dc_xf15_hi'
	27	7	41	1	0.033333333333333	0	0	'dc_xf16_hi'
	33	10	42	1	0.033333333333333	0	0	'dc_xf17_hi'
	33	10	43	1	0.033333333333333	0	0	'dc_xf18_hi'
	33	10	44	1	0.033333333333333	0	0	'dc_xf19_hi'
	29	31	45	1	0.02	0	0	'dc_xf2_series'
	29	9	45	1	0.013333333333333	0	0	'dc_xf2_common'
	34	11	46	1	0.033333333333333	0	0	'dc_xf20_hi'
	35	12	47	1	0.033333333333333	0	0	'dc_xf21_hi'
	35	12	48	1	0.033333333333333	0	0	'dc_xf22_hi'
	35	12	49	1	0.033333333333333	0	0	'dc_xf23_hi'
	35	12	50	1	0.033333333333333	0	0	'dc_xf24_hi'
	35	12	51	1	0.033333333333333	0	0	'dc_xf25_hi'
	35	12	52	1	0.033333333333333	0	0	'dc_xf26_hi'
	36	13	53	1	0.033333333333333	0	0	'dc_xf27_hi'
	38	15	54	1	0.033333333333333	0	0	'dc_xf28_hi'
	41	18	55	1	0.033333333333333	0	0	'dc_xf29_hi'
	29	32	56	1	0.02	0	0	'dc_xf3_series'
	29	9	56	1	0.013333333333333	0	0	'dc_xf3_common'
	42	19	57	1	0.033333333333333	0	0	'dc_xf30_hi'
	42	19	58	1	0.033333333333333	0	0	'dc_xf31_hi'
	42	19	59	1	0.033333333333333	0	0	'dc_xf32_hi'
	42	19	60	1	0.033333333333333	0	0	'dc_xf33_hi'
	42	19	61	1	0.033333333333333	0	0	'dc_xf34_hi'
	42	19	62	1	0.033333333333333	0	0	'dc_xf35_hi'
	43	20	63	1	0.033333333333333	0	0	'dc_xf36_hi'
	43	20	64	1	0.033333333333333	0	0	'dc_xf37_hi'
	43	20	65	1	0.033333333333333	0	0	'dc_xf38_hi'
	30	31	66	1	0.02	0	0	'dc_xf4_series'
	30	9	66	1	0.013333333333333	0	0	'dc_xf4_common'
	30	32	67	1	0.02	0	0	'dc_xf5_series'
	30	9	67	1	0.013333333333333	0	0	'dc_xf5_common'
	21	1	68	1	0.033333333333333	0	0	'dc_xf6_hi'
	21	1	69	1	0.033333333333333	0	0	'dc_xf7_hi'
	21	1	70	1	0.033333333333333	0	0	'dc_xf8_hi'
	21	1	71	1	0.033333333333333	0	0	'dc_xf9_hi'
};


%% branch_gmd data
%column_names% hi_bus lo_bus gmd_br_hi gmd_br_lo gmd_k gmd_br_series gmd_br_common baseMVA type config
mpc.branch_gmd = {
	1	2	-1	-1	0	-1	-1	100	'line'	'none'
	6	10	-1	-1	0	-1	-1	100	'line'	'none'
	7	8	-1	-1	0	-1	-1	100	'line'	'none'
	8	9	-1	-1	0	-1	-1	100	'line'	'none'
	8	10	-1	-1	0	-1	-1	100	'line'	'none'
	11	13	-1	-1	0	-1	-1	100	'line'	'none'
	11	14	-1	-1	0	-1	-1	100	'line'	'none'
	1	3	-1	-1	0	-1	-1	100	'line'	'none'
	12	13	-1	-1	0	-1	-1	100	'line'	'none'
	12	23	-1	-1	0	-1	-1	100	'line'	'none'
	13	23	-1	-1	0	-1	-1	100	'line'	'none'
	14	16	-1	-1	0	-1	-1	100	'line'	'none'
	15	16	-1	-1	0	-1	-1	100	'line'	'none'
	15	21	-1	-1	0	-1	-1	100	'line'	'none'
	15	21	-1	-1	0	-1	-1	100	'line'	'none'
	15	24	-1	-1	0	-1	-1	100	'line'	'none'
	16	17	-1	-1	0	-1	-1	100	'line'	'none'
	16	19	-1	-1	0	-1	-1	100	'line'	'none'
	1	5	-1	-1	0	-1	-1	100	'line'	'none'
	17	18	-1	-1	0	-1	-1	100	'line'	'none'
	17	22	-1	-1	0	-1	-1	100	'line'	'none'
	18	21	-1	-1	0	-1	-1	100	'line'	'none'
	18	21	-1	-1	0	-1	-1	100	'line'	'none'
	19	20	-1	-1	0	-1	-1	100	'line'	'none'
	19	20	-1	-1	0	-1	-1	100	'line'	'none'
	20	23	-1	-1	0	-1	-1	100	'line'	'none'
	20	23	-1	-1	0	-1	-1	100	'line'	'none'
	21	22	-1	-1	0	-1	-1	100	'line'	'none'
	2	4	-1	-1	0	-1	-1	100	'line'	'none'
	2	6	-1	-1	0	-1	-1	100	'line'	'none'
	3	9	-1	-1	0	-1	-1	100	'line'	'none'
	4	9	-1	-1	0	-1	-1	100	'line'	'none'
	5	10	-1	-1	0	-1	-1	100	'line'	'none'
	24	3	-1	-1	1.8	34	35	100	'xfmr'	'gwye-gwye-auto'
	2	29	36	-1	1.8	-1	-1	100	'xfmr'	'gwye-delta'
	2	30	37	-1	1.8	-1	-1	100	'xfmr'	'gwye-delta'
	2	31	38	-1	1.8	-1	-1	100	'xfmr'	'gwye-delta'
	2	32	39	-1	1.8	-1	-1	100	'xfmr'	'gwye-delta'
	7	33	40	-1	1.8	-1	-1	100	'xfmr'	'gwye-delta'
	7	34	41	-1	1.8	-1	-1	100	'xfmr'	'gwye-delta'
	7	35	42	-1	1.8	-1	-1	100	'xfmr'	'gwye-delta'
	13	36	43	-1	1.8	-1	-1	100	'xfmr'	'gwye-delta'
	13	37	44	-1	1.8	-1	-1	100	'xfmr'	'gwye-delta'
	13	38	45	-1	1.8	-1	-1	100	'xfmr'	'gwye-delta'
	11	9	-1	-1	1.8	46	47	100	'xfmr'	'gwye-gwye-auto'
	14	39	48	-1	1.8	-1	-1	100	'xfmr'	'gwye-delta'
	15	40	49	-1	1.8	-1	-1	100	'xfmr'	'gwye-delta'
	15	41	50	-1	1.8	-1	-1	100	'xfmr'	'gwye-delta'
	15	42	51	-1	1.8	-1	-1	100	'xfmr'	'gwye-delta'
	15	43	52	-1	1.8	-1	-1	100	'xfmr'	'gwye-delta'
	15	44	53	-1	1.8	-1	-1	100	'xfmr'	'gwye-delta'
	15	45	54	-1	1.8	-1	-1	100	'xfmr'	'gwye-delta'
	16	46	55	-1	1.8	-1	-1	100	'xfmr'	'gwye-delta'
	18	47	56	-1	1.8	-1	-1	100	'xfmr'	'gwye-delta'
	21	48	57	-1	1.8	-1	-1	100	'xfmr'	'gwye-delta'
	12	9	-1	-1	1.8	58	59	100	'xfmr'	'gwye-gwye-auto'
	22	49	60	-1	1.8	-1	-1	100	'xfmr'	'gwye-delta'
	22	50	61	-1	1.8	-1	-1	100	'xfmr'	'gwye-delta'
	22	51	62	-1	1.8	-1	-1	100	'xfmr'	'gwye-delta'
	22	52	63	-1	1.8	-1	-1	100	'xfmr'	'gwye-delta'
	22	53	64	-1	1.8	-1	-1	100	'xfmr'	'gwye-delta'
	22	54	65	-1	1.8	-1	-1	100	'xfmr'	'gwye-delta'
	23	55	66	-1	1.8	-1	-1	100	'xfmr'	'gwye-delta'
	23	56	67	-1	1.8	-1	-1	100	'xfmr'	'gwye-delta'
	23	57	68	-1	1.8	-1	-1	100	'xfmr'	'gwye-delta'
	11	10	-1	-1	1.8	69	70	100	'xfmr'	'gwye-gwye-auto'
	12	10	-1	-1	1.8	71	72	100	'xfmr'	'gwye-gwye-auto'
	1	25	73	-1	1.8	-1	-1	100	'xfmr'	'gwye-delta'
	1	26	74	-1	1.8	-1	-1	100	'xfmr'	'gwye-delta'
	1	27	75	-1	1.8	-1	-1	100	'xfmr'	'gwye-delta'
	1	28	76	-1	1.8	-1	-1	100	'xfmr'	'gwye-delta'
};


%% branch_thermal data
%column_names% xfmr temperature_ambient hotspot_instant_limit hotspot_avg_limit hotspot_rated topoil_time_const topoil_rated topoil_init topoil_initialized hotspot_coeff
mpc.branch_thermal = {
	0	25	-1	-1	-1	-1	-1	-1	-1	-1
	0	25	-1	-1	-1	-1	-1	-1	-1	-1
	0	25	-1	-1	-1	-1	-1	-1	-1	-1
	0	25	-1	-1	-1	-1	-1	-1	-1	-1
	0	25	-1	-1	-1	-1	-1	-1	-1	-1
	0	25	-1	-1	-1	-1	-1	-1	-1	-1
	0	25	-1	-1	-1	-1	-1	-1	-1	-1
	0	25	-1	-1	-1	-1	-1	-1	-1	-1
	0	25	-1	-1	-1	-1	-1	-1	-1	-1
	0	25	-1	-1	-1	-1	-1	-1	-1	-1
	0	25	-1	-1	-1	-1	-1	-1	-1	-1
	0	25	-1	-1	-1	-1	-1	-1	-1	-1
	0	25	-1	-1	-1	-1	-1	-1	-1	-1
	0	25	-1	-1	-1	-1	-1	-1	-1	-1
	0	25	-1	-1	-1	-1	-1	-1	-1	-1
	0	25	-1	-1	-1	-1	-1	-1	-1	-1
	0	25	-1	-1	-1	-1	-1	-1	-1	-1
	0	25	-1	-1	-1	-1	-1	-1	-1	-1
	0	25	-1	-1	-1	-1	-1	-1	-1	-1
	0	25	-1	-1	-1	-1	-1	-1	-1	-1
	0	25	-1	-1	-1	-1	-1	-1	-1	-1
	0	25	-1	-1	-1	-1	-1	-1	-1	-1
	0	25	-1	-1	-1	-1	-1	-1	-1	-1
	0	25	-1	-1	-1	-1	-1	-1	-1	-1
	0	25	-1	-1	-1	-1	-1	-1	-1	-1
	0	25	-1	-1	-1	-1	-1	-1	-1	-1
	0	25	-1	-1	-1	-1	-1	-1	-1	-1
	0	25	-1	-1	-1	-1	-1	-1	-1	-1
	0	25	-1	-1	-1	-1	-1	-1	-1	-1
	0	25	-1	-1	-1	-1	-1	-1	-1	-1
	0	25	-1	-1	-1	-1	-1	-1	-1	-1
	0	25	-1	-1	-1	-1	-1	-1	-1	-1
	0	25	-1	-1	-1	-1	-1	-1	-1	-1
	1	25	280	240	150	71	75	0	1	0.63
	1	25	280	240	150	71	75	0	1	0.63
	1	25	280	240	150	71	75	0	1	0.63
	1	25	280	240	150	71	75	0	1	0.63
	1	25	280	240	150	71	75	0	1	0.63
	1	25	280	240	150	71	75	0	1	0.63
	1	25	280	240	150	71	75	0	1	0.63
	1	25	280	240	150	71	75	0	1	0.63
	1	25	280	240	150	71	75	0	1	0.63
	1	25	280	240	150	71	75	0	1	0.63
	1	25	280	240	150	71	75	0	1	0.63
	1	25	280	240	150	71	75	0	1	0.63
	1	25	280	240	150	71	75	0	1	0.63
	1	25	280	240	150	71	75	0	1	0.63
	1	25	280	240	150	71	75	0	1	0.63
	1	25	280	240	150	71	75	0	1	0.63
	1	25	280	240	150	71	75	0	1	0.63
	1	25	280	240	150	71	75	0	1	0.63
	1	25	280	240	150	71	75	0	1	0.63
	1	25	280	240	150	71	75	0	1	0.63
	1	25	280	240	150	71	75	0	1	0.63
	1	25	280	240	150	71	75	0	1	0.63
	1	25	280	240	150	71	75	0	1	0.63
	1	25	280	240	150	71	75	0	1	0.63
	1	25	280	240	150	71	75	0	1	0.63
	1	25	280	240	150	71	75	0	1	0.63
	1	25	280	240	150	71	75	0	1	0.63
	1	25	280	240	150	71	75	0	1	0.63
	1	25	280	240	150	71	75	0	1	0.63
	1	25	280	240	150	71	75	0	1	0.63
	1	25	280	240	150	71	75	0	1	0.63
	1	25	280	240	150	71	75	0	1	0.63
	1	25	280	240	150	71	75	0	1	0.63
	1	25	280	240	150	71	75	0	1	0.63
	1	25	280	240	150	71	75	0	1	0.63
	1	25	280	240	150	71	75	0	1	0.63
	1	25	280	240	150	71	75	0	1	0.63
	1	25	280	240	150	71	75	0	1	0.63
};


%% bus_gmd data
%column_names% lat lon
mpc.bus_gmd = {
	40.4397	-78.802528470588
	40.4397	-78.726794823529
	40.903653873874	-79.611332235294
	40.700674054054	-79.256930352941
	40.700674054054	-79.067596235294
	41.077636576577	-78.613194352941
	40.497694234234	-78.196659294118
	40.526691351351	-78.499593882353
	41.034140900901	-78.991862588235
	41.034140900901	-78.991862588235
	41.034140900901	-78.991862588235
	41.034140900901	-78.991862588235
	41.222622162162	-78.348126588235
	41.483596216216	-79.256930352941
	41.454599099099	-79.711332235294
	41.628581801802	-79.749199058824
	41.860558738739	-79.938533176471
	42.005544324324	-79.862799529412
	41.773567387387	-79.446264470588
	42.005544324324	-78.953995764706
	41.94755009009	-79.521998117647
	42.411503963964	-78.726794823529
	42.021532792793	-78.651061176471
	40.903653873874	-79.611332235294
	40.4397	-78.802528470588
	40.4397	-78.802528470588
	40.4397	-78.802528470588
	40.4397	-78.802528470588
	40.4397	-78.726794823529
	40.4397	-78.726794823529
	40.4397	-78.726794823529
	40.4397	-78.726794823529
	40.497694234234	-78.196659294118
	40.497694234234	-78.196659294118
	40.497694234234	-78.196659294118
	41.222622162162	-78.348126588235
	41.222622162162	-78.348126588235
	41.222622162162	-78.348126588235
	41.483596216216	-79.256930352941
	41.454599099099	-79.711332235294
	41.454599099099	-79.711332235294
	41.454599099099	-79.711332235294
	41.454599099099	-79.711332235294
	41.454599099099	-79.711332235294
	41.454599099099	-79.711332235294
	41.628581801802	-79.749199058824
	42.005544324324	-79.862799529412
	41.94755009009	-79.521998117647
	42.411503963964	-78.726794823529
	42.411503963964	-78.726794823529
	42.411503963964	-78.726794823529
	42.411503963964	-78.726794823529
	42.411503963964	-78.726794823529
	42.411503963964	-78.726794823529
	42.021532792793	-78.651061176471
	42.021532792793	-78.651061176471
	42.021532792793	-78.651061176471
};


%% time_elapsed
%column_names% seconds
mpc.time_elapsed = 10.0;


%% thermal caps
% thermal_cap_x0 ([per unit])
%column_names% A B C D E F G H I J K
mpc.thermal_cap_x0 = [
	0.23033 0.25000 0.26438 0.27960 0.30000 0.31967 0.33942 0.36153 0.38444 0.40000 0.43894
];
% thermal_cap_y0 ([percent per unit])
%column_names% A B C D E F G H I J K
mpc.thermal_cap_y0 = [
	100.0 93.94 90.0 85.42 80.0 74.73 70.0 64.94 59.97 56.92 50.0 
];
% Values are from Fig.2. of https://arxiv.org/pdf/1701.01469.pdf paper


%%-----  SourceID Data  -----%%

%% bus_sourceid data
%column_names% bus_sid
mpc.bus_sourceid = [
	'1 ';
	'2 ';
	'3 ';
	'4 ';
	'5 ';
	'6 ';
	'7 ';
	'8 ';
	'9 ';
	'10 ';
	'11 ';
	'12 ';
	'13 ';
	'14 ';
	'15 ';
	'16 ';
	'17 ';
	'18 ';
	'19 ';
	'20 ';
	'21 ';
	'22 ';
	'23 ';
	'24 ';
	'25 ';
	'26 ';
	'27 ';
	'28 ';
	'29 ';
	'30 ';
	'31 ';
	'32 ';
	'33 ';
	'34 ';
	'35 ';
	'36 ';
	'37 ';
	'38 ';
	'39 ';
	'40 ';
	'41 ';
	'42 ';
	'43 ';
	'44 ';
	'45 ';
	'46 ';
	'47 ';
	'48 ';
	'49 ';
	'50 ';
	'51 ';
	'52 ';
	'53 ';
	'54 ';
	'55 ';
	'56 ';
	'57 ';
];


%% gen_sourceid data
%column_names% bus_i gen_sid
mpc.gen_sourceid = [
	25 '1 ';
	34 '2 ';
	35 '3 ';
	36 '4 ';
	37 '5 ';
	38 '6 ';
	39 '7 ';
	40 '8 ';
	41 '9 ';
	42 '10 ';
	43 '11 ';
	26 '12 ';
	44 '13 ';
	45 '14 ';
	46 '15 ';
	47 '16 ';
	48 '17 ';
	49 '18 ';
	50 '19 ';
	51 '20 ';
	52 '21 ';
	53 '22 ';
	27 '23 ';
	54 '24 ';
	55 '25 ';
	56 '26 ';
	57 '27 ';
	28 '28 ';
	29 '29 ';
	30 '30 ';
	31 '31 ';
	32 '32 ';
	33 '33 ';
];


%% branch_sourceid data
%column_names% fbus tbus branch_sid
mpc.branch_sourceid = [
	1 2 '1 ';
	6 10 '2 ';
	7 8 '3 ';
	8 9 '4 ';
	8 10 '5 ';
	11 13 '6 ';
	11 14 '7 ';
	1 3 '8 ';
	12 13 '9 ';
	12 23 '10 ';
	13 23 '11 ';
	14 16 '12 ';
	15 16 '13 ';
	15 21 '14 ';
	15 21 '15 ';
	15 24 '16 ';
	16 17 '17 ';
	16 19 '18 ';
	1 5 '19 ';
	17 18 '20 ';
	17 22 '21 ';
	18 21 '22 ';
	18 21 '23 ';
	19 20 '24 ';
	19 20 '25 ';
	20 23 '26 ';
	20 23 '27 ';
	21 22 '28 ';
	2 4 '29 ';
	2 6 '30 ';
	3 9 '31 ';
	4 9 '32 ';
	5 10 '33 ';
	3 24 '34 ';
	2 29 '35 ';
	2 30 '36 ';
	2 31 '37 ';
	2 32 '38 ';
	7 33 '39 ';
	7 34 '40 ';
	7 35 '41 ';
	13 36 '41 ';
	13 37 '42 ';
	13 38 '43 ';
	9 11 '44 ';
	14 39 '45 ';
	15 40 '46 ';
	15 41 '47 ';
	15 42 '48 ';
	15 43 '49 ';
	15 44 '50 ';
	15 45 '51 ';
	16 46 '52 ';
	18 47 '53 ';
	21 48 '54 ';
	9 12 '55 ';
	22 49 '56 ';
	22 50 '57 ';
	22 51 '58 ';
	22 52 '59 ';
	22 53 '60 ';
	22 54 '61 ';
	23 55 '62 ';
	23 56 '63 ';
	23 57 '64 ';
	10 11 '65 ';
	10 12 '66 ';
	1 25 '67 ';
	1 6 '68 ';
	1 27 '69 ';
	1 28 '70 ';
];


