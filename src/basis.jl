########################
## Exports #############
########################

export BasisFunctions, evaluate
export Monomials, tensor_basis

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

struct Partial{S, N} <: BasisOperator 
    component::Int

    Partial{S, N}(component::Int) = component ≤ S ? new{S, N}(component) : error("Component of partial $(component) must be ≤ $(S)")
end

struct Gradient{S} <: BasisOperator end


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

function (monomial::Monomial)(x::AbstractVector{F})::F where {F}
    result = zero(F)
    for (x, p) in zip(x, monomials.powers)
        result += x ^ p
    end
    result
end

function (partial::Partial{S, N})(monomial::Monomial{S}) where {S, N}
    power = @inbounds monomial.powers[partial.component]

    if power < N
        Monomial(@SVector zeros(S))
    else
        coefficient = one(UInt)

        for i in (power-N):power
            coefficient *= i
        end

        Monomial(@SVector zeros(S))
    end
end

"""
    tensor_basis(S, order)

Constructs a set of basis vectors for an `S` dimensional space, covering all permuations of monomials up to the given order.
"""
function tensor_basis(S, order)
    orders = Iterators.product([1:order for _ in 1:S]...)
    BasisFunctions(vec(collect(map((x) -> Monomial{S}(SVector{S, UInt}(x)), orders))))
end
