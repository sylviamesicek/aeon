########################
## Exports #############
########################

export AnalyticFunction, AnalyticGradient, AnalyticHessian, Basis, gradient, hessian
export TransformedFunction, TransformedGradient, TransformedHessian, transform
export Monomial, MonomialGradient, MonomialHessian, tensor_basis
export Gaussian

########################
## Core Types ##########
########################

"""
Represents an analytic function, linear combinations of which approximate numerical functions across nodes.
"""
abstract type AnalyticFunction end

(func::AnalyticFunction)(x) = error("Value of Analytic Function $(typeof(func)) is undefined")
gradient(func::AnalyticFunction) = error("Gradient of Analytic Function $(typeof(func)) is undefined")
hessian(func::AnalyticFunction) = error("Hessian of Analytic Function $(typeof(func)) is undefined")

abstract type AnalyticGradient end

(grad::AnalyticGradient)(x) = error("Value of Analytic Gradient $(typeof(grad)) is undefined")

abstract type AnalyticHessian end

(hess::AnalyticHessian)(x) = error("Value of Analytic Hessian $(typeof(hess)) is undefined")

"""
Represents a set of analytic functions to serve as a basis for a function space. 
"""
struct Basis{B <: AnalyticFunction}
    funcs::Vector{B}
end

Base.length(basis::Basis) = length(basis.funcs)
Base.getindex(basis::Basis, i) = getindex(basis.funcs, i)

########################
## Transformations #####
########################

"""
Represents a coordinate transformation applied to a basis function.
"""
struct TransformedFunction{T<: Transform, F<: AnalyticFunction} <: AnalyticFunction
    transform::T
    func::F
end

transform(transform::T, func::F) where {T<: Transform, F<:AnalyticFunction} = TransformedFunction{T, F}(transform, func)

(func::TransformedFunction)(x) = func.func(func.transform(x))

struct TransformedGradient{T<: Transform, G<:AnalyticGradient} <: AnalyticGradient
    transform::T
    grad::G
end

transform(transform::T, grad::G) where {T<: Transform, G<:AnalyticGradient} = TransformedGradient{T, G}(transform, grad)

(func::TransformedGradient)(x) = jacobian(func.transform, x) * func.basis(func.transform(x))

struct TransformedHessian{T<: Transform, H<:AnalyticHessian} <: AnalyticHessian
    transform::T
    hess::H
end

transform(transform::T, hess::H) where {T<: Transform, H<:AnalyticHessian} = TransformedHessian{T, H}(transform, hess)

(hess::TransformedHessian)(x) = jacobian(hess.transform, x) * jacobian(hess.transform, x) * hess.hess(func.transform(x))

########################
## Monomial ############
########################

"""
Represents a monomial term, ie a combination of powers of each coordinate. For instance 1, x, xy, x², etc.
Each monomial term is an N-Vector containing the power of that component. 
For instance x z^2 is represented as [1, 0, 2].
"""
struct Monomial{S} <: AnalyticFunction
    powers::SVector{S, UInt}
end

function (monomial::Monomial)(x::AbstractVector{F})::F where  {F}
    result = zero(F)

    for (x, p) in zip(x, monomials.powers)
        result += x ^ p
    end

    result
end

"""
    tensor_basis(S, order)

Constructs a set of basis vectors for an `S` dimensional space, covering all permuations of monomials up to the given order.
"""
function tensor_basis(S, order)
    orders = Iterators.product([1:order for _ in 1:S]...)
    BasisFunctions(vec(collect(map((x) -> Monomial{S}(SVector{S, UInt}(x)), orders))))
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

###########################
## Gausian ################
###########################

struct Gaussian{F} <: AnalyticFunction
    amplitude::F
    simga::F
end

(gauss::Gaussian)(x) = e^(-dot(x, x)/(gauss.simga * gauss.simga))
