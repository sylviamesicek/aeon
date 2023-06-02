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

    for (x, p) in zip(x, monomials.powers)
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

function (monomial::MonomialGradient{S})(x::AbstractVector{F}) where {S, F}
    grad::SVector{S, F} = @SVector zeros(S)

    for i in 1:S
        for (x, p) in zip(x, monomials.powers)
            sub = convert(UInt, p == i && p != 0)
            grad[i] += x^(p - sub)
        end
    end 

    grad .*= monomial.powers

    grad
end

"""
`Hessian` applied to a monomial function.
"""
struct MonomialHessian{S} <: AnalyticHessian
    coefficients::SMatrix{S, S, UInt}
    powers::SVector{S, UInt}
end

function hessian(monomial::Monomial{S}) where {S}
    coefficients::SMatrix{S, S, F} = @SMatrix zeros(S, S)

    for i in 1:S
        for j in 1:S
            if i == j
                power = monomial.powers[i]
                coefficients[i, i] = power > 1 ? power * (power - 1) : 0
            else
                coefficients[i, j] = monomial.powers[i] * monomial.powers[j]
            end
        end
    end

    MonomialHessian(coefficients, monomial.powers)
end

function (hess::MonomialHessian{S})(x::AbstractVector{F}) where {S, F}
    hessian::SMatrix{S, S, F} = @SMatrix zeros(S, S)

    for i in 1:S
        for j in 1:S
            for (x, p) in zip(x, hess.powers)
                sub = convert(UInt, p == i && p != 0) + convert(UInt, p == j && p != 0)
                hessian[i, j] += x^(p - sub)
            end
        end 
    end 

    hessian .*= hes.coefficients

    hessian
end