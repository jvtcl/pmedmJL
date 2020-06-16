module Pmedm

using DataFrames
using SparseArrays
using LinearAlgebra
using Kronecker
using Optim
using Random
using Distributions

export pmd, compute_allocation, pmedm_solve,
simulate_probabilities

mutable struct pmd

    # inputs
    gl::Array{String,2}
    pX::Array{Int64,2}
    wt::Array{Real,1}
    Y1::Array{Real,2}
    Y2::Array{Real,2}
    V1::Array{Real,2}
    V2::Array{Real,2}
    N::Real
    n::Real
    A1::Array{Int64,2}
    A2::Array{Bool,2}
    X::SparseMatrixCSC{Int64, Int64}
    Y_vec::Vector{Float64}
    V_vec::Vector{Float64}
	sV::Diagonal
    λ::Vector{Float64}
    q::Vector{Float64}
    p::Vector{Float64}
	H::Array{Float64,2}

    function pmd(gl, pX, wt, Y1, Y2, V1, V2, N, n)

        ## geographies
		A0 = Int64[]
        for G in unique(gl[:,2])
         isG = [Int(occursin(G, g)) for g in gl[:,2]]
         append!(A0, isG)
        end

        A1 = reshape(A0, (size(gl)[1], length(unique(gl[:,2]))))'

        A2 = Matrix(I, size(gl)[1], size(gl)[1])

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

		## Hessian
		# placeholder until solver is run
		H = zeros(length(λ), length(λ))

        new(gl, pX, wt, Y1, Y2, V1, V2, N, n, A1, A2, X,
		    Y_vec, V_vec, sV, λ, q, p, H)
    end

end

compute_allocation = function(pmd::pmd)

    qXl = pmd.q .* exp.(-pmd.X * pmd.λ)
	qXl / sum(qXl)

end

pmedm_solve = function(pmd::pmd)

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

	q = pmd.q
	X = pmd.X
	λ = pmd.λ
	Y_vec = pmd.Y_vec
	sV = pmd.sV

	opt = optimize(f, g!, h!, λ, NewtonTrustRegion(),
	               Optim.Options(show_trace=true, iterations = 200))

	pmd.λ = Optim.minimizer(opt)
	pmd.p = compute_allocation(pmd)
	pmd.H = h!(pmd.H, pmd.λ)

	return pmd

end

simulate_probabilities = function(pmd::pmd, nsim = 100, seed = 0)

	covλ = inv(pmd.H)

	simλ = []

	Random.seed!(seed)
	mvn = MvNormal(pmd.λ, Matrix(Hermitian(covλ / pmd.N)))

	simλ = rand(mvn, nsim)

	## simulate p
	psim = []

	for s in 1:nsim
		pmds = pmd
		pmds.λ = simλ[:,s]
	    ps = compute_allocation(pmds)
	    ps = ps * pmds.N
	    append!(psim, ps)
	end

	reshape(psim, :, nsim);

end

end
