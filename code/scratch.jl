#%%
using Base
using CSV
using DataFrames
using SparseArrays
using LinearAlgebra
using SuiteSparse
using Statistics
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
q = repeat(wt, size(A1)[2]);
q = reshape(q, n, size(A1)[2]);
q = q / sum(q);
q = vec(q')
#%%

#%% Vectorize geo. constraints (Y) and normalize
# Y_vec = vec(vcat(Y1, Y2)) / N
Y_vec = vcat(vec(Y1), vec(Y2)) / N; # fix
#%%

#%% Vectorize variances and normalize
# V_vec = vec(vcat(V1, V2)) * (n / N^2)
V_vec = vcat(vec(V1), vec(V2)) * (n / N^2); # fix
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
λ = repeat([0.], length(Y_vec));
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

end
#%%

#%% test it
# @time phat = compute_allocation(q, X, λ)
# @time phat = reshape(phat, size(pX)[1], size(A2)[1])
#%%

#%% compute the block group constraint estimates
# @time Yhat2 = (N * phat)' * pX;
#%%

#%% compute the tract constraint estimates
# Yhat1 = phat_trt' * pX
# phat_trt = (phat * N) * A1';
#%%

#%% Vectorize constraint estimates
# Yhat = vec(vcat(Yhat1, Yhat2))
#%%

#%% Assemble results
# Ype = DataFrame(Y = Y_vec * N, Yhat = Yhat, V = V_vec * (N^2/n))
#%%

#%% Primal function (scratch)
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

    penalty = (e^2 / (2. * v))

    ent = ((n / N) * (w / d) * log((w/d)))

    pe = (-1. * ent) - penalty

    return pe

end
#%%

#%% TEST - apply PE function
# pe = penalized_entropy.(Ype.Y, Ype.Yhat, n, N, Ype.V)
#%%

#%% Objective function
neg_pe = function(λ)

    phat = compute_allocation(q, X, λ)
    # phat = reshape(phat, size(pX)[1], size(A2)[1])
    phat = reshape(phat, size(A2)[1], size(pX)[1])'; # FIX - row-major reshaping - matches

    Yhat2 = (N * phat)' * pX

    phat_trt = (phat * N) * A1'
    Yhat1 = phat_trt' * pX

    # Yhat = vec(vcat(Yhat1, Yhat2))
    Yhat = vcat(vec(Yhat1), vec(Yhat2)); # FIX

    Ype = DataFrame(Y = Y_vec * N, Yhat = Yhat, V = V_vec * (N^2/n))

    pe = penalized_entropy.(Ype.Y, Ype.Yhat, n, N, Ype.V)

    -1. * mean(pe)

end
#%%

#%%
# @time neg_pe(λ)
#%%

using Optim
# @time opt = optimize(neg_pe, zeros(length(Y_vec)),
#                     Optim.Options(show_trace=true))

@time opt = optimize(neg_pe, zeros(length(Y_vec)), BFGS(), autodiff = :forward,
            Optim.Options(show_trace=true, iterations = 200))

# update lambda
λ = Optim.minimizer(opt)

neg_pe(λ)

#%%check results
phat = compute_allocation(q, X, λ)
phat = reshape(phat, size(A2)[1], size(pX)[1])'; # FIX - row-major reshaping - matches

Yhat2 = (N * phat)' * pX

phat_trt = (phat * N) * A1'
Yhat1 = phat_trt' * pX

Yhat = vcat(vec(Yhat1), vec(Yhat2)); # FIX - matches

Ype = DataFrame(Y = Y_vec * N, Yhat = Yhat, V = V_vec * (N^2/n))

#90% MOEs
Ype.MOE_lower = Ype.Y - (sqrt.(Ype.V) * 1.645);
Ype.MOE_upper = Ype.Y + (sqrt.(Ype.V) * 1.645);

# Proportion of contstraints falling outside 90% MOE
sum((Ype.Yhat .< Ype.MOE_lower) + (Ype.Yhat .> Ype.MOE_upper) .>= 1) / nrow(Ype)

#%%

###########
## with BlackBoxOptim
# using BlackBoxOptim
#
# res = bboptimize(neg_pe; InitialCandidate = repeat([0.], length(Y_vec)), NumDimensions = length(λ), method = :dxnes)
#
# λf = best_candidate(res);
# neg_pe(λf)

# neg_pe(repeat([0.], length(Y_vec)))

# ####### TROUBLESHOOTING (OLD) #########
#
# ### solution from python
# λf = [ 0.53389188, -0.0379973, -0.50200761, -0.16321265, -0.13225007,
#         0.57743544, -0.03988062,  0.22363727,  0.23495304, -0.604144  ,
#        -0.11663491,  0.26081404, -1.22108342,  0.31515081,  1.43999659,
#         0.38795166, -0.42577341, -0.61241939, -0.92714886, -1.7774103 ,
#         0.98007359,  1.8342036 , -0.67483984, -0.87041146,  1.38211506,
#        -1.02380534,  0.89157858, -0.38641539,  1.32246777,  2.00724413,
#        -2.15040901, -0.21542829,  1.92205921,  0.81691693, -2.77884608,
#         1.35971006, -1.1360607 ,  0.85969393,  0.49921946,  0.93573072,
#         0.20624954, -2.26602294,  1.85283508, -0.98020062, -1.47673084,
#        -0.66216718,  0.54559164,  0.38439201,  0.38497734,  2.09420423,
#        -0.6503332 , -1.95243673]
# neg_pe(λf) # doesn't match

# compute_allocation(q, X, λf)

#%% TEST COMPUTE ALLOCATION (all good)
# a0 = exp.(-X * λf); # matches
#
# # use `.*` to broadcast vector (element-wise vs. object-wise multiplication)
# a = a0 .* q; # matches
#
# b = q' * a0; # matches
#
# blah = a/b; # matches
#%%

#%% TEST RECONSTRUCT CONSTRAINTS
# phat = compute_allocation(q, X, λf); # matches
# # phat = reshape(phat, size(pX)[1], size(A2)[1]);
# phat = reshape(phat, size(A2)[1], size(pX)[1])'; # FIX - row-major reshaping - matches
#
# Yhat2 = (N * phat)' * pX; # matches
#
# phat_trt = (phat * N) * A1'; # matches
# Yhat1 = phat_trt' * pX; # matches
#
# # Yhat = vec(vcat(Yhat1, Yhat2));
# Yhat = vcat(vec(Yhat1), vec(Yhat2)); # FIX - matches
#
# Ype = DataFrame(Y = Y_vec * N, Yhat = Yhat, V = V_vec * (N^2/n)); # matches
#%%

#%% TEST - objective fn
# pe = penalized_entropy.(Ype.Y, Ype.Yhat, n, N, Ype.V);
#
# -1. * mean(pe) # matches

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
