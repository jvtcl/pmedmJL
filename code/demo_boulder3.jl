#%%
using Base
using CSV
using DataFrames
using SparseArrays
using Kronecker
using LinearAlgebra
using SuiteSparse
using Statistics
#%%

#%%
include("./pmedm.jl")
using .Pmedm
#%%

#%% read in data
# alt-shift-enter to run the cell
constraints_ind = CSV.read("data/boulder_constraints_ind_2016_person.csv")
constraints_bg = CSV.read("data/boulder_constraints_bg_2016_person.csv")
constraints_trt = CSV.read("data/boulder_constraints_trt_2016_person.csv");
constraints_ind = constraints_ind[:,2:ncol(constraints_ind)]
constraints_bg = constraints_bg[:,2:ncol(constraints_bg)]
constraints_trt = constraints_trt[:,2:ncol(constraints_trt)];
#%%

#%% geo lookup
geo_lookup = CSV.read("data/boulder_geo_lookup.csv")[:,["bg", "trt"]];
geo_lookup.bg = string.(geo_lookup.bg)
geo_lookup.trt = string.(geo_lookup.trt);
#%%

#%% Ensure tract IDs between `constraints_trt` and `geo_lookup` are consistent
tix = indexin(unique(geo_lookup.trt), string.(constraints_trt.GEOID));
constraints_trt = constraints_trt[tix,:];
#%%

#%% sanity check
sum(unique(geo_lookup.bg) .== string.(constraints_bg.GEOID)) == nrow(constraints_bg)
sum(unique(geo_lookup.trt) .== string.(constraints_trt.GEOID)) == nrow(constraints_trt)
#%%

#%% PUMS response ids
serial = collect(constraints_ind.pid);
#%%

#%% sample weights
wt = collect(constraints_ind.wt);
#%%

#%% population and sample size
# use semicolons to supress printing
N = sum(constraints_bg.POP);
n = nrow(constraints_ind);
#%%

#%% individual (PUMS) constraints
# we need to use the ∉ symbol to represent a logical "not in"
# type it as \notin + TAB
excl = ["pid", "wt"]
constraint_cols = [i ∉ excl for i in names(constraints_ind)];
pX = constraints_ind[!,constraint_cols];
pX = convert(Matrix, pX);
#%%

#%%
# pX = sparse(pX)
# pX = dropzeros(pX);
#%%

#%% geographic constraints
est_cols_bg = [!endswith(i, 's') && i != "GEOID" for i in names(constraints_bg)]
est_cols_trt = [!endswith(i, 's') && i != "GEOID" for i in names(constraints_trt)]
Y1 = convert(Matrix, constraints_trt[!,est_cols_trt])
Y2 = convert(Matrix, constraints_bg[!,est_cols_bg]);
#%%

#%% error variances
se_cols = [endswith(i, 's') for i in names(constraints_bg)]
se_cols = names(constraints_bg)[se_cols]
V1 = map(x -> x^2, convert(Matrix, constraints_trt[!,se_cols]))
V2 = map(x -> x^2, convert(Matrix, constraints_bg[!,se_cols]));
#%%

#%%
gl = Array(geo_lookup)
#%%

#%% P-MEDM Data
bou = pmd(gl, pX, wt, Y1, Y2, V1, V2, N, n);
#%%

#%% Run the P-MEDM solver
bou = pmedm_solve(bou)
#%%

#%% check results
phat = reshape(bou.p, size(bou.A2)[1], size(bou.pX)[1])'

Yhat2 = (bou.N * phat)' * bou.pX

phat_trt = (phat * bou.N) * bou.A1'
Yhat1 = phat_trt' * bou.pX

Yhat = vcat(vec(Yhat1), vec(Yhat2));

Ype = DataFrame(Y = bou.Y_vec * bou.N,
                Yhat = Yhat, V = bou.V_vec * (bou.N^2/bou.n))

#90% MOEs
Ype.MOE_lower = Ype.Y - (sqrt.(Ype.V) * 1.645);
Ype.MOE_upper = Ype.Y + (sqrt.(Ype.V) * 1.645);

# Proportion of contstraints falling outside 90% MOE
const_match = sum((Ype.Yhat .< Ype.MOE_lower) + (Ype.Yhat .> Ype.MOE_upper) .>= 1) / nrow(Ype)
#%%

#%% Simulate P-MEDM Probabilities
psim = simulate_probabilities(bou);
#%%
