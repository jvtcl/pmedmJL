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

#%% Geographic crosswalk
# I think there is a MUCH more elegant way to do this
# with dicts -- come back to this...

A1 = Int64[]

for G in unique(geo_lookup.trt)

    blah = zeros(Int64, 1, nrow(constraints_bg))

    isG = [occursin(G, g) for g in collect(geo_lookup.bg)]
    for i in findall(isG)
        blah[i] = 1
    end
    append!(A1, blah)

end

A1 = reshape(A1, nrow(constraints_bg), nrow(constraints_trt))
A1 = transpose(A1)
# A1 = sparse(A1)
# A1 = dropzeros(A1)

#%%

#%% Target unit identity matrix
A2 = Matrix(I, nrow(constraints_bg), nrow(constraints_bg))
#%%

#%%
# A1 = dropzeros(sparse(A1))
# A2 = dropzeros(sparse(A2))
#%%

#%% Solution Space
X1 = (sparse(pX') ⊗ A1)'
X2 = (sparse(pX') ⊗ A2)'
@time X = hcat(sparse(X1), sparse(X2));
# X = dropzeros(X)
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

#%%
compute_allocation = function(q, X, λ)
    qXl = exp.(-X * λ) .* q
    qXl / sum(qXl)
end
#%%

#%% Objective function (PMEDMrcpp style)
f = function(λ)

    qXl = q .* exp.(-X * λ)
    p = qXl / sum(qXl)

    Xp = X' * p
    lvl = λ' * (sV * λ);

    return (Y_vec' * λ) + log(sum(qXl)) + (0.5 * lvl)

end
#%%

#%%
# # @time neg_pe(λ)
# init_λ = zeros(length(Y_vec))

λ = zeros(length(Y_vec))
# λ = sparse(λ);

# # init_λ = repeat([0.], length(Y_vec))
# # init_λ = Y_vec + (sV * λ) - Xp;
# # init_λ = rand(length(Y_vec));
# @time f(init_λ)
#%%

## making λ sparse is the only thing that seems to help time-wise
# q = sparse(q);
# X = dropzeros(X)
@time f(λ)

#%%
# using ForwardDiff
# # @time Y_vec + (sV * λ) - Xp
# @time ForwardDiff.gradient(f, λ)
#%%

#%% Gradient function
g! = function(G, λ)
    qXl = q .* exp.(-X * λ)
    p = qXl / sum(qXl)
    Xp = X'p
    G[:] = Y_vec + (sV * λ) - Xp
end

@time g!(zeros(length(λ)), λ)
#%%

#%% Hessian
h! = function(H, λ)
    qXl = q .* exp.(-X * λ)
    p = qXl / sum(qXl)
    dp = sparse(Diagonal(p))
    H[:] = -((X'p) * (p'X)) + (X' * dp * X) + sV
end
#%%

# @time h!(Array{Float64}(undef, 1665, 1665), init_λ)

# precond = function(n)
#     spdiagm((-ones(n-1), 2*ones(n), -ones(n-1)), (-1,0,1), n, n)*(n+1)
# end
#
# precond(1665)
#
# n = 1665
# spdiagm((-ones(n-1), 2*ones(n), -ones(n-1)), (-1,0,1), n, n)*(n+1)
#
# precond(1665)
#
# spdiagm(-1 => [1,2,3,4], 1 => [4,3,2,1])

#%%
using Optim

init_λ = zeros(length(Y_vec));
# init_λ = rand(length(Y_vec))

# SOME NOTES ON THE DIFFERENT SOLVERS
#
# - TRUST REGION is extremely slow - MUCH faster with precomputed Hessian
#
# - GRADIENT DESCENT does not converge at 1000 iterations
#   but gives a reasonable solution (0.84% constraints unmatched)

@time opt = optimize(f, g!, h!, init_λ, NewtonTrustRegion(),
                    Optim.Options(show_trace=true, iterations = 200))

# @time opt = optimize(f, g!, init_λ, GradientDescent(),
                    # Optim.Options(show_trace=true, iterations = 1000))

# # SLOW!! ~3.74 hours
# @time opt = optimize(f, init_λ, BFGS(), autodiff = :forward,
#             Optim.Options(show_trace=true, iterations = 10)) # orig 100

# final lambda
λ = Optim.minimizer(opt);
#%%

# # dump results
# CSV.write("data/boulder_run1.csv", DataFrame(lambda = λ));

#
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
#
# ###########
#%% with BlackBoxOptim
# using BlackBoxOptim
# #
# res = bboptimize(f; InitCandidate = init_λ, NumDimensions = length(init_λ), MaxSteps = 10, SearchRange = (-5.,5.), method = :ProbabalisticDescent)
#%%
#
# #%%
# λf = best_candidate(res);
#
# neg_pe(λf)
# #%%
#
# #%%check results
# # phat = compute_allocation(q, X, zeros(length(Y_vec)))
# phat = compute_allocation(q, X, λf) # final
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
#
# #%%
#
# ################
#
# ## matching pmedm.cpp
# blah = zeros(size(X)[1]);
# mul!(blah, -X, sparse(λ));
# a0 = exp.(blah);
# qXl = a0 .* q;
#
# p = qXl / sum(qXl);
#
# blah2 = zeros(size(X)[2]);
# Xp = mul!(blah2, X', p);
#
# lvl = λ' * (sV * λ);
#
# fun = (Y_vec' * λ) + log(sum(qXl)) + (0.5 * lvl)
#
# grad = Y_vec + (sV * λ) - Xp
#
# #%%
#
# #%%
# function prob(λ)
#     blah = zeros(size(X)[1]);
#     mul!(blah, -X, λ);
#     a0 = exp.(blah);
#     qXl = a0 .* q;
#
#     return (qXl / sum(qXl))
# end
# #%%
#
# #%%
# function f(λ)
#
#     blah = zeros(size(X)[1]);
#     mul!(blah, -X, λ);
#     a0 = exp.(blah);
#     qXl = a0 .* q;
#
#     p = qXl / sum(qXl);
#
#     blah2 = zeros(size(X)[2]);
#     Xp = mul!(blah2, X', p)
#     lvl = λ' * (sV * λ);
#
#     return (Y_vec' * λ) + log(sum(qXl)) + (0.5 * lvl)
# end
# #%%
#
# #%%
# function g!(λ)
#
#     blah = zeros(size(X)[1]);
#     mul!(blah, -X, λ);
#     a0 = exp.(blah);
#     qXl = a0 .* q;
#
#     p = qXl / sum(qXl);
#
#     blah2 = zeros(size(X)[2]);
#     Xp = mul!(blah2, X', p);
#
#     return (Y_vec + (sV * λ) - Xp)
# end
# #%%
#
# #%%
# init_λ = sparse(zeros(length(Y_vec)))
# # p = prob(λ)
#
# # λ = sprand(1665, 1.)
# #%%
#
# #%%
# f(init_λ)
# g!(init_λ)
# #%%
#
# #%%
# # optimize(f, g!, init_λ, BFGS(), Optim.Options(show_trace=true, iterations = 5))
# # optimize(f, init_λ, BFGS(), autodiff = :forward, Optim.Options(show_trace=true, iterations = 5))
# #%%
