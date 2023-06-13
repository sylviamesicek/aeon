# Exports

export AnalyticBasis
export monomials

# Core

"""
Represents a set of analytic functions to serve as a basis for a function space. 
"""
struct AnalyticBasis{N, T, F}
    inner::Vector{F}

    AnalyticBasis{N, T}(funcs::Vector{F}) where {N, T, F <: AnalyticField{N, T, 0}} = new{N, T, F}(funcs)
end

Base.length(basis::AnalyticBasis) = length(basis.inner)
Base.eachindex(basis::AnalyticBasis) = eachindex(basis.inner)
Base.getindex(basis::AnalyticBasis, i) = getindex(basis.inner, i)

# Monomials

"""
    monomials(N, order)

Constructs a set of basis vectors for a `D` dimensional space, covering all permuations of monomials up to the given order.
"""
function monomials(::Val{N}, ::Val{T}, order) where{N, T}
    dims = ntuple(_ -> 0:order, Val(N))
    ocoords = CartesianIndices(dims)

    result = Vector{Monomial{N, T}}(undef, length(ocoords))

    for (i, x) in enumerate(ocoords)
        result[i] = Monomial{N, T}(SVector(Tuple(x)))
    end

    AnalyticBasis{N, T}(result)
end