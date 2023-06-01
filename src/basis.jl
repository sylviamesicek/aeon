########################
## Exports #############
########################

export BasisFunctions, evaluate
export Monomials, tensor_basis

########################
## Core Types ##########
########################

abstract type BasisFunction end

struct BasisFunctions{B <: BasisFunction}
    bases::Vector{B}
end

Base.length(basis::BasisFunctions) = length(basis.bases)
Base.getindex(basis::BasisFunctions, i) = getindex(basis.bases, i)

abstract type Derivative end

derivative(func::BasisFunction)::BasisFunction = error("Analytic derivative not implemented for $(typeof(func))")

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

    Monomial{S}(x) where {S} = new{S}(SVector{S, UInt}(x))
end

function (monomial::Monomial)(x::AbstractVector{F})::F where {F}
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
    BasisFunctions(vec(collect(map(Monomial{S}, orders))))
end

