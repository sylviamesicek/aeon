using Symbolics
using TensorOperations
using LinearAlgebra

###############################
## Types of Derivatives #######
###############################

function ∂(cotensor::Array{T,N}, chart::Vector{T}) where {T,N}
    result = Array{T,N + 1}(undef, size(cotensor)..., length(chart))

    for I in CartesianIndices(cotensor)
        for j in eachindex(chart)
            result[I, j] = Symbolics.derivative(cotensor[I], chart[j])
        end
    end

    return result
end

function ∂(cotensor::T, chart::Vector{T}) where {T}
    result = Array{T,1}(undef, length(chart))

    for i in eachindex(chart)
        result[i] = Symbolics.derivative(cotensor, chart[i])
    end

    return result
end

function ∇(cotensor::Array{T,N}, chart::Vector{T}, connection::Array{T,3}) where {T,N}
    result = ∂(cotensor, chart)

    for dim in 1:N
        for I in CartesianIndices(cotensor)
            for i in eachindex(chart), j in eachindex(chart)
                Inew = Base.setindex(Tuple(I), j, dim)
                result[I, i] -= connection[j, I[dim], i] * cotensor[Inew...]
            end
        end
    end

    return result
end

∇(cotensor::T, chart::Vector{T}, connection::Array{T,3}) where {T} = ∂(cotensor, chart)

function ℒ(cotensor::Array{T,N}, chart::Vector{T}, vector::Vector{T}) where {T,N}
    # Get partial derivatives of both tensor and vector
    tgrad = ∂(cotensor, chart)
    vgrad = ∂(vector, chart)

    # Accumulate result
    result = Array{T,N}(undef, size(cotensor)...)
    fill!(result, zero(T))

    for I in CartesianIndices(cotensor)
        for j in eachindex(chart)
            result[I] += vector[j] * tgrad[I, j]
        end
    end

    # Add corrections
    for dim in 1:N
        for I in CartesianIndices(cotensor)
            for j in eachindex(chart)
                Inew = Base.setindex(Tuple(I), j, dim)
                result[I] += vgrad[j, I[dim]] * cotensor[Inew...]
            end
        end
    end

    return result
end

ℒ(cotensor::T, chart::Vector{T}, vector::Vector{T}) where {T} = dot(∂(cotensor, chart), vector)

##################################
## Connection ####################
##################################

function christoffel(G::Array{T,2}, Ginv::Array{T,2}, chart::Vector{T}) where {T}
    Gpar = ∂(G, chart)

    Γ = Array{Num}(undef, 2, 2, 2)
    for I in CartesianIndices(Γ)
        (λ, μ, ν) = Tuple(I)
        Γ[λ, μ, ν] = sum([1 // 2 * Ginv[λ, σ] * (Gpar[ν, σ, μ] + Gpar[σ, μ, ν] - Gpar[μ, ν, σ]) for σ in eachindex(chart)])
    end

    return Γ
end

#################################
## Geometric Tensors ############
#################################

function ricci(Γ::Array{T,3}, chart::Vector{T}) where {T}
    Γpar = ∂(Γ, chart)

    R = Array{Num}(undef, 2, 2)
    for I in CartesianIndices(R)
        (μ, ν) = Tuple(I)
        R[μ, ν] = sum([Γpar[σ, μ, ν, σ] - Γpar[σ, μ, σ, ν] for σ in eachindex(chart)])
        R[μ, ν] += sum(Γ[σ, μ, ν] * Γ[τ, σ, τ] - Γ[σ, μ, τ] * Γ[τ, ν, σ] for σ in eachindex(chart), τ in eachindex(chart))
    end

    return R
end

function main()

    ########################
    ## Chart ###############
    ########################

    @variables ρ z
    coords = [ρ, z]

    ########################
    ## Degrees of Freedom ##
    ########################

    # Metric
    @variables g₁₁(ρ, z), g₁₂(ρ, z), g₂₂(ρ, z)
    G = [g₁₁ g₁₂; g₁₂ g₂₂]
    @variables S(ρ, z)
    λ::Num = ρ * exp(ρ * S) * sqrt(g₁₁)
    # Extrinsic Curvature
    @variables k₁₁(ρ, z), k₁₂(ρ, z), k₂₂(ρ, z)
    K = [k₁₁ k₁₂; k₁₂ g₂₂]
    @variables y(ρ, z)
    L = ρ * y + k₁₁ / g₁₁
    # Constraint variables
    @variables θ(ρ, z), z₁(ρ, z), z₂(ρ, z)
    Z = [z₁, z₂]
    # Gauge variables
    @variables α(ρ, z), β¹(ρ, z), β²(ρ, z)
    β = [β¹, β²]

    ############################
    ## Common Expressions ######
    ############################

    # Compute Metric Inverse and Determinant
    Ginv = inv(G)
    Gdet = det(G)
    # Compute Christoffel symbols and Partials
    Γ = christoffel(G, Ginv, coords)
    # Compute Ricci Tensor and trace
    R = ricci(Γ, coords)
    Rtrace = sum([R[μ, ν] * Ginv[μ, ν] for μ in 1:2, ν in 1:2])
    # Contractions of K
    Ktrace = sum([K[μ, ν] * Ginv[μ, ν] for μ in 1:2, ν in 1:2])
    Kmat = [sum(Ginv[μ, σ] * K[σ, ν] for σ in 1:2) for μ in 1:2, ν in 1:2]
    Kcon = [sum(K[σ, τ] * Ginv[σ, μ] * Ginv[τ, ν] for σ in 1:2, τ in 1:2) for μ in 1:2, ν in 1:2]
    # Contraction of Z
    Zcon = [sum(Z[j] * Ginv[i, j] for j in 1:2) for i in 1:2]
    # Derivatives of G
    Ginvpar = ∇(Ginv, coords, Γ)
    Gdetpar = ∇(Gdet, coords, Γ)
    # Derivatives of K
    Kgrad = ∇(K, coords, Γ)
    # Derivatives of λ
    λgrad = ∇(λ, coords, Γ)
    λhess = ∇(λgrad, coords, Γ)
    # Derivatives of θ
    θgrad = ∇(θ, coords, Γ)
    # Derivatives of z
    Zgrad = ∇(Z, coords, Γ)
    # Derivatives of α
    αgrad = ∇(α, coords, Γ)
    αhess = ∇(αgrad, coords, Γ)
    # Derivatives of β
    βpar = ∂(β, coords)

    ##############################
    ## Hamiltonian Constraint ####
    ##############################

    Hamiltonian::Num = 1 // 2 * (Rtrace + Ktrace^2) + Ktrace * L

    for i in 1:2, j in 1:2
        Hamiltonian -= 1 // 2 * K[i, j] * Kcon[i, j]
        Hamiltonian -= λhess[i, j] * Ginv[i, j]
    end

    ##############################
    ## Momentum Constraints ######
    ##############################

    Mterm1 = -∇(Ktrace + L, coords, Γ) - λgrad / λ * L
    Mterm2 = [sum(λgrad[j] * Kmat[j, i] for j in 1:2) for i in 1:2]
    Mterm3 = [sum(Kgrad[i, j, k] * Ginv[j, k] for j in 1:2, k in 1:2) for i in 1:2]

    Momentum::Vector{Num} = Mterm1 + Mterm2 + Mterm3

    ##############################
    ## Extrinsic Evolution #######
    ##############################

    Kterm1 = R - λhess / λ - αhess / α
    Kterm2 = (Ktrace + L) * K - 2 * [sum(K[i, k] * Kmat[k, j] for k in 1:2) for i in 1:2, j in 1:2]
    Kterm3 = [Zgrad[i, j] + Zgrad[j, i] for i in 1:2, j in 1:2] - 2 * K * θ
    LieK::Matrix{Num} = Kterm1 + Kterm2 + Kterm3

    Lterm1 = sum(-λhess[i, j] / λ * Ginv[i, j] for i in 1:2, j in 1:2)
    Lterm2 = sum(-λgrad[i] / λ * αgrad[j] / α * Ginv[i, j] for i in 1:2, j in 1:2)
    Lterm3 = L * (Ktrace + L) - 2 * L * θ
    Lterm4 = sum(2 * (λgrad[i] / λ) * Zcon[i] for i in 1:2)
    LieL = Lterm1 + Lterm2 + Lterm3 + Lterm4

    ###############################
    ## Metric Evolution ###########
    ###############################

    LieG = -2 * K
    Lieλ = -λ * L

    ###############################
    ## Constraint Evolution #######
    ###############################

    θterm1 = sum((λgrad[i] / λ - αgrad[i] / α) * Z[j] * Ginv[i, j] for i in 1:2, j in 1:2)
    θterm2 = sum(Zgrad[i, j] * Ginv[i, j] for i in 1:2, j in 1:2) - (Ktrace + L) * θ
    Lieθ = Hamiltonian + θterm1 + θterm2

    Zterm1 = [sum(-2 * K[i, j] * Z[j] for j in 1:2) for i in 1:2]
    Zterm2 = -αgrad / α * θ + θgrad
    LieZ = Momentum + Zterm1 + Zterm2

    ###############################
    ## Time derivatives ###########
    ###############################

    Gdt = α * LieG + ℒ(G, coords, β)
    λdt = α * Lieλ + ℒ(λ, coords, β)
    Kdt = α * LieK + ℒ(K, coords, β)
    Ldt = α * LieL + ℒ(L, coords, β)

    θdt = α * Lieθ + ℒ(θ, coords, β)
    Zdt = α * LieZ + ℒ(Z, coords, β)

    # Regularized variables
    Ydt = (Ldt - Kdt[1, 1] / G[1, 1] + K[1, 1] / G[1, 1]^2 * Gdt[1, 1]) / ρ
    Sdt = (λdt / λ - 1 // 2 * Gdt[1, 1] / G[1, 1]) / ρ

    ################################
    ## Gauge Evolution #############
    ################################

    # Gauge freedoms
    f = n = d = a = 1
    m = 2

    αterm1 = -α^2 * f * (Ktrace + L - m * θ)
    αterm2 = dot(β, αgrad)
    αdt = αterm1 + αterm2

    λGterm = [sum((λgrad[j] / λ + Gdetpar[j] / Gdet) * Ginv[i, j] for j in 1:2) for i in 1:2]
    βterm1 = [sum(a * αgrad[j] / α * Ginv[i, j] for j in 1:2) for i in 1:2]
    βterm2 = 2 * n * (λGterm - Zcon) - d * λGterm
    βterm3 = [n * sum(Ginvpar[i, j, j] for j in 1:2) for i in 1:2]
    βterm4 = [-(2 * n - d) * Ginv[1, 1] / ρ, 0]
    βterm5 = [sum(β[j] * βpar[i, j] for j in 1:2) for i in 1:2]
    βdt = -α^2 * (βterm1 + βterm2 + βterm3 + βterm4) + βterm5

    # return Kdt[1, 2]
end

# main()

@variables x, tvar, tvard

tfunc(x) = missing
tfuncd(x) = missing

@register_symbolic tfunc(x)
@register_symbolic tfuncd(x)

Symbolics.derivative(::typeof(tfunc), args::NTuple{1,Any}, ::Val{1}) = tfuncd(args[1])

function test()
    expr = tfunc(x) + Symbolics.derivative(tfunc(x), x)
    expr = substitute(expr, Dict([tfunc(x) => tvar, tfuncd(x) => tvard]))

    return simplify(expr)
end

test()