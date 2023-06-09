####################
## Exports #########
####################

export AnalyticBasis, monomials

####################
## Core ############
####################

"""
Represents a set of analytic functions to serve as a basis for a function space. 
"""
struct AnalyticBasis{N, T, F, R}
    funcs::Vector{F}

    AnalyticBasis(funcs::Vector{AnalyticField{N, T, R}}) where {N, T, R} = new{N, T, eltype(funcs), R}(funcs)
end

Base.length(basis::AnalyticBasis) = length(basis.funcs)
Base.eachindex(basis::AnalyticBasis) = eachindex(basis.funcs)
Base.getindex(basis::AnalyticBasis, i) = getindex(basis.funcs, i)

####################
## Monomial Bases ##
####################

"""
    monomials(N, order)

Constructs a set of basis vectors for a `D` dimensional space, covering all permuations of monomials up to the given order.
"""
function monomials(N, T, order)
    orders = Iterators.product([0:order for _ in 1:N]...)
    orders = vec(collect(map(Monomial{T}, orders)))

    AnalyticBasis{N, T, Monomial{N, T}, 0}(orders)
end