#%%
using Base
using CSV
using DataFrames
using SparseArrays
using LinearAlgebra
using SuiteSparse
using Statistics
using Kronecker
#%%

#%% read in data
# alt-shift-enter to run the cell
constraints_ind = CSV.read("data/toy_constraints_ind.csv")
constraints_bg = CSV.read("data/toy_constraints_bg.csv")
constraints_trt = CSV.read("data/toy_constraints_trt.csv")
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
#%%

#%% Target unit identity matrix
A2 = Matrix(I, nrow(constraints_bg), nrow(constraints_bg))
#%%

#%% Solution Space
X1 = (sparse(pX') ⊗ A1)'
X2 = (sparse(pX') ⊗ A2)'
@time X = hcat(sparse(X1), sparse(X2));
#%%

#%% Design Weights
q = repeat(wt, size(A1)[2]);
q = reshape(q, n, size(A1)[2]);
q = q / sum(q);
q = vec(q')
#%%

#%% Vectorize geo. constraints (Y) and normalize
Y_vec = vcat(vec(Y1), vec(Y2)) / N; # fix
#%%

#%% Vectorize variances and normalize
V_vec = vcat(vec(V1), vec(V2)) * (n / N^2); # fix
#%%

#%% Diagonal matrix of variances
sV = Diagonal(V_vec)
#%%

#%% Compute the P_MEDM probabilities from q, X, λ
compute_allocation = function(q, X, λ)

    qXl = q .* exp.(-X * λ)
    p = qXl / sum(qXl)

end
#%%

#%% Objective Function
f = function(λ)

    qXl = exp.(-X * λ) .* q
    p = qXl / sum(qXl)

    Xp = X' * p
    lvl = λ' * (sV * λ);

    return (Y_vec' * λ) + log(sum(qXl)) + (0.5 * lvl)

end
#%%

#%%
g! = function(G, λ)
    qXl = q .* exp.(-X * λ)
    p = qXl / sum(qXl)
    Xp = X'p
    G[:] = Y_vec + (sV * λ) - Xp
end
#%%

#%% Hessian
h! = function(H, λ)
    qXl = q .* exp.(-X * λ)
    p = qXl / sum(qXl)
    dp = sparse(Diagonal(p))
    H[:] = -((X'p) * (p'X)) + (X' * dp * X) + sV
end
#%%

#%%
using Optim

init_λ = zeros(length(Y_vec))

opt = optimize(f, g!, h!, init_λ, NewtonTrustRegion(),
               Optim.Options(show_trace=true, iterations = 200));

#%%

#%%
# # with preconditioner
# # updated version from github
# precond(n::Number) = Optim.InverseDiagonal(diag(spdiagm(-1 => -ones(n-1), 0 => 2*ones(n), 1 => -ones(n-1)) * (n+1)))
#
# opt = optimize(f, g!, h!, init_λ, ConjugateGradient(P = precond(length(init_λ))),
#                Optim.Options(show_trace=true, iterations = 200, g_tol = 1e-4));
#%%

#%%check results
# update lambda
λ = Optim.minimizer(opt)

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

A1x = Array()


[[Int(occursin(G, g)) for g in geo_lookup[:,2]] for G in unique(geo_lookup[:,2])]


A1x = Int64[]

for G in unique(geo_lookup[:,2])
    isG = [Int(occursin(G, g)) for g in geo_lookup[:,2]]
    append!(A1x, isG)
end

A1x = reshape(A1x, (length(unique(geo_lookup[:,2])), nrow(geo_lookup)))
