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

#%%
# blah = pmedmobj(pX)
# blah.pX
#%%

#%%
blah = pmedmobj(Y1, Y2, N)
#%%
