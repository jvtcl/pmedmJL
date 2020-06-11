module Pmedm

using DataFrames
using SparseArrays
using Optim

export f, g!, h!, pmedmobj

f = function(λ)

    qXl = q .* exp.(-X * λ)
    p = qXl / sum(qXl)

    Xp = X' * p
    lvl = λ' * (sV * λ);

    return (Y_vec' * λ) + log(sum(qXl)) + (0.5 * lvl)

end

g! = function(G, λ)
    qXl = q .* exp.(-X * λ)
    p = qXl / sum(qXl)
    Xp = X'p
    G[:] = Y_vec + (sV * λ) - Xp
end

h! = function(H, λ)
    qXl = q .* exp.(-X * λ)
    p = qXl / sum(qXl)
    dp = spdiagm(0 => p)
    H[:] = -((X'p) * (p'X)) + (X' * dp * X) + sV
end

mutable struct pmedmobj

    # inputs
    gl::Array{String,2}
    pX::Array{Int64,2}
    Y1::Array{Int64,2}
    Y2::Array{Int64,2}
    V1::Float64{Int64,2}
    V2::Array{Int64,2}
    N::Int64
    n::Int64
    A1::Array{Int64}
    A2::Array{Bool, 2}
    X::SparseMatrixCSC{Int64, Int64}
    Y_vec::Vector{Float64}
    # V_vec::Vector{Float64}
    sV::Array{Float64,2}
    λ::Vector{Float64}
    q::Vector{Float64}
    p::Vector{Float64}

    function pmedmobj(pX, Y1, Y2, V1, V2, N, n)

        ## geographies
        for G in unique(gl[:,2])
            isG = [Int(occursin(G, g)) for g in gl[:,2]]
            append!(A1, isG)
        end

        A1 = reshape(A1, (length(unique(gl[:,2])), nrow(gl)))

        A2 = Matrix(I, nrow(constraints_bg), nrow(constraints_bg))

        ## model matrix
        X1 = (sparse(pX') ⊗ A1)'
        X2 = (sparse(pX') ⊗ A2)'
        X = hcat(sparse(X1), sparse(X2))

        ## constraints
        Y_vec = vcat(vec(Y1), vec(Y2)) / N

        ## variances
        V_vec = vcat(vec(V1), vec(V2)) * (n / N^2)
        sV = Diagonal(V_vec)

        ## design weights
        q = repeat(wt, size(A1)[2])
        q = reshape(q, n, size(A1)[2])
        q = q / sum(q)
        q = vec(q')

        ## coefficients
        λ = zeros(length(Y_vec))

        ## probabilities
        p = zeros(length(q))

        new(gl, pX, Y1, Y2, V1, V2, N, n, A1, A2, X, Y_vec, sV, λ, q, p)
    end

end

end
