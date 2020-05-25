#%%
using Base
using CSV
using DataFrames
using SparseArrays
using LinearAlgebra
using SuiteSparse
#%%

#%% read in data
# alt-shift-enter to run the cell
constraints_ind = CSV.read("data/toy_constraints_ind.csv")
constraints_bg = CSV.read("data/toy_constraints_bg.csv")
constraints_trt = CSV.read("data/toy_constraints_trt.csv")
#%%

#%%
# # use `collect` to get column values
# collect(constraints_bg.GEOID)
#
# # or if we wanted to use numerical indices
# collect(constraints_bg[!,1])
# # this subsets rows
# constraints_bg[1:2,:]
#%%

#%% build geo lookup
# apply string conversion to bg GEOIDs
bg_id = string.(collect(constraints_bg[!,1]))
trt_id = [s[1:2] for s in bg_id]
geo_lookup = DataFrame(bg = bg_id, trt = trt_id)
#%%

#%% PUMS response ids
serial = collect(constraints_ind.SERIAL)
#%%

#%% sample weights
wt = collect(constraints_ind.PERWT)
#%%

#%% population and sample size
# use semicolons to supress printing
N = sum(constraints_bg.POP);
n = nrow(constraints_ind);
#%%

#%% individual (PUMS) constraints
# we need to use the ∉ symbol to represent a logical "not in"
# type it as \notin + TAB
excl = ["SERIAL", "PERWT"]
constraint_cols = [i ∉ excl for i in names(constraints_ind)];
pX = constraints_ind[!,constraint_cols];
pX = convert(Matrix, pX);
# pX = sparse(pX);
#%%

#%% geographic constraints
est_cols = [!endswith(i, 's') && i != "GEOID" for i in names(constraints_bg)]
Y1 = convert(Matrix, constraints_trt[!,est_cols])
Y2 = convert(Matrix, constraints_bg[!,est_cols]);
#%%

#%% error variances
se_cols = [endswith(i, 's') for i in names(constraints_bg)]
se_cols = names(constraints_bg)[se_cols]
V1 = map(x -> x^2, convert(Matrix, constraints_trt[!,se_cols]))
V2 = map(x -> x^2, convert(Matrix, constraints_bg[!,se_cols]));
#%%

#%% Geographic crosswalk
# I think there is a MUCH more elegant way to do this
# with dicts -- come back to this...

A1 = []

for G in unique(geo_lookup.trt)

    blah = zeros(Int8, 1, nrow(constraints_bg))

    isG = [occursin(G, g) for g in collect(geo_lookup.bg)]
    for i in findall(isG)
        blah[i] = 1
    end
    append!(A1, blah)

end

A1 = reshape(A1, nrow(constraints_bg), nrow(constraints_trt))
A1 = transpose(A1)
# A1 = sparse(A1)

#%%

#%% Target unit identity matrix
# A2 = Matrix{Int8}(I, nrow(constraints_bg), nrow(constraints_bg))
A2 = Matrix(I, nrow(constraints_bg), nrow(constraints_bg))
# A2 = sparse(A2)
#%%

#%% Solution Space
X1 = kron(transpose(pX), A1)
X2 = kron(transpose(pX), A2)
X = transpose(vcat(X1, X2))
#%%

#%% Design Weights
q = repeat(wt, size(A1)[2])
q = reshape(q, n, size(A1)[2])
q = q / sum(q);
q = vec(q)
#%%

#%% Vectorize geo. constraints (Y) and normalize
Y_vec = vec(vcat(Y1, Y2)) / N
#%%

#%% Vectorize variances and normalize
V_vec = vec(vcat(V1, V2)) * (n / N^2)
#%%

#%% Diagonal matrix of variances
# sV = diagm(V_vec)
sV = Diagonal(V_vec)
#%%

#%% sparsify inputs
# X = sparse(X)
# sV = sparse(sV)
#%%

#%% Initial lambdas (test)
λ = repeat([0], length(Y_vec));
#%%

#%% SCRATCH
# a0 = exp.(-X * λ);
#
# # use `.*` to broadcast vector (element-wise vs. object-wise multiplication)
# a = a0 .* q;
#
# b = q' * a0;
#
# a/b;
#%%

#%% Compute the P_MEDM probabilities from q, X, λ
compute_allocation = function(q, X, λ)

    a0 = exp.(-X * λ)

    a = a0 .* q;

    b = q' * a0

    a/b


end;
#%%

#%% test it
@time phat = compute_allocation(q, X, λ)
@time phat = reshape(phat, size(pX)[1], size(A2)[1])
#%%

#%% compute the block group constraint estimates
@time Yhat2 = (N * phat)' * pX;
#%%

#%% compute the tract constraint estimates
phat_trt = (phat * N) * A1';
Yhat1 = phat_trt' * pX
#%%

#%% Vectorize constraint estimates
Yhat = vec(vcat(Yhat1, Yhat2))
#%%

#%% Assemble results
Ype = DataFrame(Y = Y_vec * N, Yhat = Yhat, V = V_vec * (N^2/n))
#%%

#%% Primal Funciton (scratch)
# w = Ype[1,:].Y
# d = Ype[1,:].Yhat
# v = Ype[1,:].V
#
# e = d - w
#
# penalty = (e^2 / (2 * v))
#
# ent = ((n / N) * (w / d) * log((w/d)))
#
# pe = (-1 * ent) - penalty
#%%

#%% Primal Function
penalized_entropy = function(w, d, n, N, v)

    e = d - w

    penalty = (e^2 / (2 * v))

    ent = ((n / N) * (w / d) * log((w/d)))

    pe = (-1 * ent) - penalty

    return pe

end
#%%

#%% TEST - apply PE function
penalized_entropy.(Ype.Y, Ype.Yhat, n, N, Ype.V)
#%%

#### MISC ####
# # applying functions
# square(x) = x^2
# blah = [1,2,3]
# square.(blah)
#
# # subsetting
# constraints_bg[["GEOID", "CONST1"]] # columns
# constraints_bg[!,[1,4]] # columns by index
# constraints_bg[1:2,] # rows
#
# size(constraints_trt)
# size(constraints_bg)
