export Monomial, MonomialGradient, MonomialHessian

# Core

"""
Represents a monomial term, ie a combination of powers of each coordinate. For instance 1, x, xy, x², etc.
Each monomial term is an N-Vector containing the power of that component. 
For instance x z^2 is represented as [1, 0, 2].
"""
struct Monomial{N, T} <: AnalyticField{N, T, 0} 
    powers::SVector{N, Int}
end

Monomial{T}(powers::SVector{N, Int}) where {N, T} = Monomial{N, T}(powers)
Monomial{T}(powers::Int...) where T = Monomial{T}(SVector(powers...))

function (monomials::Monomial{N, T})(x::SVector{N, T}) where {N, T}
    result = zero(F)

    for (x, p) in zip(x, monomial.powers)
        result += x ^ p
    end

    result
end

# Gradient

"""
The gradient of a monomial term.
"""
struct MonomialGradient{N, T} <: AnalyticField{N, T, 1}
    powers::SVector{N, Int}
end

(::AnalyticDerivative{N, T})(field::Monomial{N, T}) where {N, T} = MonomialGradient{N, T}(field.powers)

function power(monomial::MonomialGradient{N, T}, x::SVector{N, T}, i::Int) where {N, T}
    result = zero(F)

    for (x, p) in zip(x, monomial.powers)
        sub = convert(Int, p == i && p != 0)
        result += x^(p - sub)
    end

    result
end

function (monomial::MonomialGradient{N, T})(x::SVector{N, T}) where {N, T}
    sacollect(SVector{N, T}, power(monomial, x, i) for i in 1:S)
end

# Hessian

"""
The hessian of a monomial term.
"""
struct MonomialHessian{N, T, L} <: AnalyticField{N, T, 2}
    coefficients::SMatrix{N, N, Int, L}
    powers::SVector{N, Int}
end

(::AnalyticCurvature{N, T})(monomial::Monomial{N, T}) where {N, T} = MonomialHessian{N, T}(sacollect(SMatrix{N, N, Int, N*N}, coefficient(monomial, i, j) for i in 1:S, j in 1:S), monomial.powers)

function coefficient(monomial::MonomialHessian, i::Int, j::Int)
    if i == j
        power = monomial.powers[i]
        return power > 1 ? power * (power - 1) : 0
    else
        return monomial.powers[i] * monomial.powers[j]
    end
end

function power(monomial::MonomialHessian{N, T}, x::SVector{N, T}, i::Int, j::Int) where {N, T}
    result = zero(T)
    for (x, p) in zip(x, monomial.powers)
        sub = convert(Int, p == i && p != 0) + convert(Int, p == j && p != 0)
        result += x^(p - sub)
    end
    result
end

function (monomial::MonomialHessian{N, T})(x::SVector{N, T}) where {N, T}
    sacollect(SMatrix{N, N, T, N*N}, power(monomial, x, i, j) for i in 1:N, j in 1:N) .* monomial.coefficients
end

