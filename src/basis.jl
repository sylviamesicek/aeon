####################
## Exports #########
####################

export Basis, tensor_basis

####################
## Core Types ######
####################

export Basis

"""
Represents a set of analytic functions to serve as a basis for a function space. 
"""
struct Basis{B <: AnalyticFunction}
    funcs::Vector{B}
end

Base.length(basis::Basis) = length(basis.funcs)
Base.getindex(basis::Basis, i) = getindex(basis.funcs, i)

####################
## Monomial Bases ##
####################

"""
    tensor_basis(S, order)

Constructs a set of basis vectors for an `S` dimensional space, covering all permuations of monomials up to the given order.
"""
function tensor_basis(S, order)
    orders = Iterators.product([1:order for _ in 1:S]...)
    BasisFunctions(vec(collect(map((x) -> Monomial{S}(SVector{S, UInt}(x)), orders))))
end