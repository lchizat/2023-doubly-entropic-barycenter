"""
Dual objective of entropy regularized optimal transport (EROT) for 
    INPUT (same conventions in other functions below):
    - u, v   : potentials (vectors of size n, m)
    - p, q   : marginals (nonnegative vectors suming to one of size n, m)
    - C      : cost matrix of size n x m
    - lambda : regularization parameter (positive scalar)
"""
function dual_objective(u, v, p, q, C, lambda)
    sum(u .* p) + sum(v .* q) + lambda  * (1 - sum(exp.((u .+ log.(p) .+ v' .+ log.(q)' .- C)/lambda)))
end

"Perform one Sinkhorn iteration starting from u0 with cost C and regularization eta (stabilized)"
function sinkhorn_iter(u0, p, q, C, lambda)
    temp = u0 .- C  
    vest = - maximum(temp , dims = 1) # to regularize log-sum-exp
    v    = vest - lambda * log.( sum( exp.((vest .+ temp)/lambda .+ log.(p)), dims = 1))
    marg_x = sum( exp.((v .+ temp )/lambda .+log.(q)'), dims = 2)
    err  = sum(p.* abs.(marg_x .- 1)) # norm ℓ1 of the gradient
    u    = u0 - lambda * log.( marg_x )
    return u, err
end

"Regularized Nonlinear Acceleration of the sequence xs = [x0,x1,..,x_k] and ys satisfying y=g(x), with regularization λ."
function RNA(xs, gxs, reg)
    U = gxs - xs
    k = size(U,2)
    K = U'*U
    nK = norm(K)
    nK > 0.0 && (K = K/nK) # avoid division by 0
    z = (K + reg*I)\ones(k)
    c = z/sum(z)
    return sum(c' .* gxs , dims = 2)
end

"""
Sinkhorn with Regularized Nonlinear Acceleration (online version)
    INPUT
    - p, q   : marginals (nonnegative vectors suming to one of size n, m)
    - C      : cost matrix of size n x m
    - lambda : regularization parameter (positive scalar)
    - tol    : tolerance (expressed as norm ℓ1 of the gradient of dual objective in u)
    OUTPUT
    - u, v   : the quasi-optimal dual potentials
    - errs   : evolution of the errors along the iterations
    - pots   : an array with the evolution of the potential u along the iterations
"""
function sinkhorn_RNA(p, q, C, lambda ; tol = 1e-10, RNA_order = 5, RNA_reg = 1e-10, maxiter = 10000)
    n = size(C,1)
    u, gu = zeros(n), zeros(n)
    us, gus = zeros(n,RNA_order), zeros(n,RNA_order)
    errs = []
    #pots = zeros(n,maxiter)
    i = 1

    # accelerated algorithm
    while ( i == 1 || errs[i-1] > tol) &&  i < maxiter
        u = RNA(us[:,1:min(i-1,RNA_order)], gus[:,1:min(i-1,RNA_order)], RNA_reg) # works for empty array but tricky
        #pots[:,i] = u
        RNA_reg = RNA_reg / 1.005
        us = cat(u, us[:,1:end-1], dims = 2)
        gu, err = sinkhorn_iter(u, p, q, C, lambda )
        gus = cat(gu, gus[:,1:end-1], dims=2)
        push!(errs, err)
        i += 1
    end

    # compute other potential
    vest = - maximum( u .+ lambda*log.(p) .- C , dims = 1) # to regularize log-sum-exp
    v    = vest - lambda*log.( mean( exp.((vest .+ u .+ lambda*log.(p) .- C)/lambda), dims = 1))
    return u[:], v[:], errs#, pots
end



