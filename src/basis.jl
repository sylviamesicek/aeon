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
struct Basis{Dim, Func <: AnalyticFunction}
    funcs::SVector{Dim, Func}
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
    orders = Iterators.product([0:order for _ in 1:S]...)
    orders = vec(collect(map((x) -> Monomial{S}(SVector{S, UInt}(x)), orders)))

    Basis{(order + 1)^S, Monomial{S}}(orders)
end