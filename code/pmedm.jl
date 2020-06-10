module Pmedm

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

    Y1::Array{Int64,2}
    Y2::Array{Int64,2}
    N::Int64
    Y_vec::Vector{Float64}

    function pmedmobj(Y1, Y2, N)
        Y_vec = vcat(vec(Y1), vec(Y2)) / N
        new(Y1, Y2, N, Y_vec)
    end

end

end
