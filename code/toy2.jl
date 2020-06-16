# cd("../")
# pwd()

#%%
include("./pmedm.jl")
using .Pmedm
#%%

#%%
using CSV
using DataFrames
constraints_ind = CSV.read("data/toy_constraints_ind.csv")
constraints_bg = CSV.read("data/toy_constraints_bg.csv")
constraints_trt = CSV.read("data/toy_constraints_trt.csv")
#%%

#%% build geo lookup
# apply string conversion to bg GEOIDs
bg_id = string.(collect(constraints_bg[!,1]))
trt_id = [s[1:2] for s in bg_id]
geo_lookup = DataFrame(bg = bg_id, trt = trt_id)
# geo_lookup = Array(geo_lookup)
#%%

#%% sample weights
wt = collect(constraints_ind.PERWT)
#%%

#%% population and sample size
N = sum(constraints_bg.POP);
n = nrow(constraints_ind);
#%%

#%%
excl = ["SERIAL", "PERWT"]
constraint_cols = [i ∉ excl for i in names(constraints_ind)];
pX = constraints_ind[!,constraint_cols];
pX = convert(Matrix, pX);
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
reg = pmd(gl, pX, wt, Y1, Y2, V1, V2, N, n);
#%%

#%%
reg = pmedm_solve(reg)
#%%

#%%
phat = reshape(reg.p, size(reg.A2)[1], size(reg.pX)[1])'

Yhat2 = (reg.N * phat)' * reg.pX

phat_trt = (phat * reg.N) * reg.A1'
Yhat1 = phat_trt' * reg.pX

Yhat = vcat(vec(Yhat1), vec(Yhat2));

Ype = DataFrame(Y = reg.Y_vec * reg.N,
                Yhat = Yhat, V = reg.V_vec * (reg.N^2/reg.n))

#90% MOEs
Ype.MOE_lower = Ype.Y - (sqrt.(Ype.V) * 1.645);
Ype.MOE_upper = Ype.Y + (sqrt.(Ype.V) * 1.645);

# Proportion of contstraints falling outside 90% MOE
const_match = sum((Ype.Yhat .< Ype.MOE_lower) + (Ype.Yhat .> Ype.MOE_upper) .>= 1) / nrow(Ype)
println(const_match)
#%%

#%% Simulate P-MEDM Probabilities
psim = simulate_probabilities(reg);
#%%
