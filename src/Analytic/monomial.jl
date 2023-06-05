############################
## Exports #################
############################

export Monomial, MonomialGradient, MonomialHessian

############################
## Monomial ################
############################

"""
Represents a monomial term, ie a combination of powers of each coordinate. For instance 1, x, xy, x², etc.
Each monomial term is an N-Vector containing the power of that component. 
For instance x z^2 is represented as [1, 0, 2].
"""
struct Monomial{S} <: AnalyticFunction
    powers::SVector{S, UInt}
end

function (monomial::Monomial)(x::AbstractVector{F}) where  {F}
    result = zero(F)

    for (x, p) in zip(x, monomial.powers)
        result += x ^ p
    end

    result
end

"""
`Gradient` applied to a monomial function.
"""
struct MonomialGradient{S} <: AnalyticGradient
    powers::SVector{S, UInt}
end

function gradient(monomial::Monomial)
    MonomialGradient(monomial.powers)
end

function power(monomial::MonomialGradient{S}, x::AbstractVector{F}, i::Int) where {S, F}
    result = zero(F)

    for (x, p) in zip(x, monomial.powers)
        sub = convert(UInt, p == i && p != 0)
        result += x^(p - sub)
    end

    result
end

function (monomial::MonomialGradient{S})(x::AbstractVector{F}) where {S, F}
    # Iterate through the powers, and collect this into a staticly sized SVector gradient
    SVector{S, F}([power(monomial, x, i) for i in 1:S]) .* monomial.powers
end

"""
`Hessian` applied to a monomial function.
"""
struct MonomialHessian{S} <: AnalyticHessian
    coefficients::SMatrix{S, S, UInt}
    powers::SVector{S, UInt}
end

function coefficient(monomial::MonomialHessian{S}, i::Int, j::Int) where {S}
    if i == j
        power = monomial.powers[i]
        return power > 1 ? power * (power - 1) : 0
    else
        return monomial.powers[i] * monomial.powers[j]
    end
end

function hessian(monomial::Monomial{S}) where {S}
    MonomialHessian(StaticArrays.sacollect(SMatrix{S, S, F}, coefficient(monomial, i, j) for i in 1:S, j in 1:S), monomial.powers)
end

function power(monomial::MonomialHessian{S}, x::AbstractVector{F}, i::Int, j::Int) where {S, F}
    result = zero(F)
    for (x, p) in zip(x, monomial.powers)
        sub = convert(UInt, p == i && p != 0) + convert(UInt, p == j && p != 0)
        result += x^(p - sub)
    end
    result
end

function (monomial::MonomialHessian{S})(x::AbstractVector{F}) where {S, F}
    SMatrix{S, S, F}([power(monomial, x, i, j) for i in 1:S, j in 1:S]) .* monomial.coefficients
end