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
A2 = Matrix(I, nrow(constraints_bg), nrow(constraints_bg))
#%%

#%% Solution Space
X1 = (sparse(pX') ⊗ A1)'
X2 = (sparse(pX') ⊗ A2)'
@time X = hcat(sparse(X1), sparse(X2));
#%%

#%% clear intermediates
X1 = nothing
X2 = nothing
pXt = nothing
#%%

#%% Design Weights
q = repeat(wt, size(A1)[2]);
q = reshape(q, n, size(A1)[2]);
q = q / sum(q);
q = vec(q')
#%%

#%% Vectorize geo. constraints (Y) and normalize
Y_vec = vcat(vec(Y1), vec(Y2)) / N;
#%%

#%% Vectorize variances and normalize
V_vec = vcat(vec(V1), vec(V2)) * (n / N^2); # fix
#%%

#%% Diagonal matrix of variances
sV = Diagonal(V_vec)
#%%

#%% Initial lambdas (test)
λ = zeros(length(Y_vec));

# # test values
# λ = sprandn(1665, 1.);
#%%

#%% SCRATCH

# blah = zeros(size(X)[1])
# mul!(blah, -X, λ);
#
# # # this works for `X * λ` but is super slow
# # @time blah = [sum(λ[-X[i,:].nzind]) for i in X.rowval]
#
# a0 = exp.(blah);
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

    blah = zeros(size(X)[1])
    mul!(blah, -X, sparse(λ));
    a0 = exp.(blah);

    a = a0 .* q;

    b = q' * a0

    a/b

end
#%%

#%% test it
# X = Matrix(X)
@time phat = compute_allocation(q, X, λ);
@time phat = reshape(phat, size(pX)[1], size(A2)[1])
#%%

#%% compute the block group constraint estimates
@time Yhat2 = (N * phat)' * sparse(pX);
#%%

#%% compute the tract constraint estimates
phat_trt = (phat * N) * A1'
@time Yhat1 = phat_trt' * pX;
#%%

#%% Vectorize constraint estimates
Yhat = vec(vcat(Yhat1, Yhat2))
#%%

#%% Assemble results
Ype = DataFrame(Y = Y_vec * N, Yhat = Yhat, V = V_vec * (N^2/n));
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
pe = penalized_entropy.(Ype.Y, Ype.Yhat, n, N, Ype.V);
#%%

#%% Objective function
neg_pe = function(λ)

    phat = compute_allocation(q, X, λ);
    # phat = reshape(phat, size(pX)[1], size(A2)[1])
    phat = reshape(phat, size(A2)[1], size(pX)[1])'; # FIX - row-major reshaping - matches

    Yhat2 = (N * phat)' * pX

    phat_trt = (phat * N) * A1'
    Yhat1 = phat_trt' * pX

    # Yhat = vec(vcat(Yhat1, Yhat2))
    Yhat = vcat(vec(Yhat1), vec(Yhat2)); # FIX

    Ype = DataFrame(Y = Y_vec * N, Yhat = Yhat, V = V_vec * (N^2/n))

    pe = penalized_entropy.(Ype.Y, Ype.Yhat, n, N, Ype.V)

    # pe[isnan.(pe)] .= 0

    # -1. * mean(pe)
    -1. * mean(filter(!isnan, pe))

end
#%%

#%%
@time neg_pe(λ)
#%%

#%%
using Optim
# @time opt = optimize(neg_pe, zeros(length(Y_vec)), BFGS(),
                    # Optim.Options(show_trace=true, iterations = 10))

@time opt = optimize(neg_pe, zeros(length(Y_vec)), LBFGS(), autodiff = :forward,
            Optim.Options(show_trace=true, iterations = 5))
#%%

#%%
# update lambda
λ = Optim.minimizer(opt)

neg_pe(λ)

# #%%check results
# phat = compute_allocation(q, X, λ)
# phat = reshape(phat, size(A2)[1], size(pX)[1])'; # FIX - row-major reshaping - matches
#
# Yhat2 = (N * phat)' * pX
#
# phat_trt = (phat * N) * A1'
# Yhat1 = phat_trt' * pX
#
# Yhat = vcat(vec(Yhat1), vec(Yhat2)); # FIX - matches
#
# Ype = DataFrame(Y = Y_vec * N, Yhat = Yhat, V = V_vec * (N^2/n))
#
# #90% MOEs
# Ype.MOE_lower = Ype.Y - (sqrt.(Ype.V) * 1.645);
# Ype.MOE_upper = Ype.Y + (sqrt.(Ype.V) * 1.645);
#
# # Proportion of contstraints falling outside 90% MOE
# sum((Ype.Yhat .< Ype.MOE_lower) + (Ype.Yhat .> Ype.MOE_upper) .>= 1) / nrow(Ype)

#%%

###########
#%% with BlackBoxOptim
using BlackBoxOptim
#
res = bboptimize(neg_pe; InitialCandidate = λ, NumDimensions = length(λ), MaxSteps = 10, method = :ProbabalisticDescent)

λf = best_candidate(res);

neg_pe(λf)
#%%

#%%check results
# phat = compute_allocation(q, X, zeros(length(Y_vec)))
phat = compute_allocation(q, X, λf) # final
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
