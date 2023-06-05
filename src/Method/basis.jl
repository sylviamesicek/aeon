####################
## Exports #########
####################

export Basis, tensor_basis

####################
## Core Types ######
####################

export Basis, tensor_basis

"""
Represents a set of analytic functions to serve as a basis for a function space. 
"""
struct Basis{F <: AnalyticFunction}
    funcs::Vector{F}
end

Base.length(basis::Basis) = length(basis.funcs)
Base.eachindex(basis::Basis) = eachindex(basis.funcs)
Base.getindex(basis::Basis, i) = getindex(basis.funcs, i)

####################
## Monomial Bases ##
####################

"""
    tensor_basis(D, order)

Constructs a set of basis vectors for a `D` dimensional space, covering all permuations of monomials up to the given order.
"""
function tensor_basis(D, order)
    orders = Iterators.product([0:order for _ in 1:D]...)
    orders = vec(collect(map((x) -> Monomial{D}(SVector{D, UInt}(x)), orders)))

    Basis{Monomial{D}}(orders)
end