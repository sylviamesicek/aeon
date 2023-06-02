########################
## Exports #############
########################

export BasisFunction, BasisOperator, BasisFunctions
export Partial
export Monomial, tensor_basis

########################
## Core Types ##########
########################

"""
Represents an analytic function, linear combinations of which approximate numerical functions across nodes.
"""
abstract type BasisFunction end

(basis::BasisFunction)(x) = error("Application of Basis Function of type $(typeof(basis)) is undefined")

"""
Represents an analytic operator which may be applied to a basis function
"""
abstract type BasisOperator end

(basis::BasisOperator)(_::BasisFunction) = error("Application of Basis Operator of type $(typeof(basis)) is undefined")

"""
Transforms the result of an operator at a point, using the given transformation.
"""
transform_operator(operator::BasisOperator,::Transform, _, _) = error("Transformation of Basis Operator $(typeof(operator)) is undefined")

"""
Represents a set of basis functions. 
"""
struct BasisFunctions{B <: BasisFunction}
    bases::Vector{B}
end

Base.length(basis::BasisFunctions) = length(basis.bases)
Base.getindex(basis::BasisFunctions, i) = getindex(basis.bases, i)

########################
## Derivative ##########
########################

"""
The `Gradient` operation, which performed on a scalar field yields a vector valued gradient.
"""
struct Gradient{S} <: BasisOperator end

transform_operator(::Gradient, trans::Transform, x, value) = jacobian(trans, x) * value

"""
The `Hessian` operation, which, when performed on a scalar field, yields a matrix valued hessian matrix.
"""
struct Hessian{S} <: BasisOperator end

transform_operator(::Hessian, trans::Transform, x, value) = jacobian(trans, x) * jacobian(trans, x) * value

########################
## Monomial ############
########################

"""
Represents a monomial term, ie a combination of powers of each coordinate. For instance 1, x, xy, x², etc.
Each monomial term is an N-Vector containing the power of that component. 
For instance x z^2 is represented as [1, 0, 2].
"""
struct Monomial{S} <: BasisFunction
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


struct MonomialGradient{S} <: BasisFunction
    powers::SVector{S, UInt}
end

function (::Gradient{S})(monomial::Monomial{S}) where {S}
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

struct MonomialHessian{S} <: BasisFunction
    coefficients::SMatrix{S, S, UInt, S*S}
    powers::SVector{S, UInt}
end

function (::Hessian{S})(monomial::Monomial{S}) where {S}
    coefficients::SMatrix{S, S, F, S * S} = @SMatrix zeros(S, S)

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

function (hes::MonomialHessian{S})(x::AbstractVector{F}) where {S, F}
    hessian::SMatrix{S, S, F, S * S} = @SMatrix zeros(S)

    for i in 1:S
        for j in 1:S
            for (x, p) in zip(x, hes.powers)
                sub = convert(UInt, p == i && p != 0) + convert(UInt, p == j && p != 0)
                hessian[i, j] += x^(p - sub)
            end
        end 
    end 

    hessian .*= hes.coefficients

    hessian
end