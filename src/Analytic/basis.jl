# Exports

export ABasis
export monomials

# Core

"""
Represents a set of analytic functions to serve as a basis for a function space. 
"""
struct ABasis{N, T, F}
    inner::Vector{F}

    ABasis{N, T}(funcs::Vector{F}) where {N, T, F <: AFunction{N, T}} = new{N, T, F}(funcs)
end

Base.length(basis::ABasis) = length(basis.inner)
Base.eachindex(basis::ABasis) = eachindex(basis.inner)
Base.getindex(basis::ABasis, i) = getindex(basis.inner, i)

# Monomials

"""
    monomials(N, order)

Constructs a set of basis vectors for a `D` dimensional space, covering all permuations of monomials up to the given order.
"""
function monomials(::Val{N}, ::Val{T}, order) where{N, T}
    dims = ntuple(_ -> 0:order, Val(N))
    ocoords = CartesianIndices(dims)

    result = Vector{AMonomial{N, T}}(undef, length(ocoords))

    for (i, x) in enumerate(ocoords)
        result[i] = AMonomial{N, T}(SVector(Tuple(x)))
    end

    ABasis{N, T}(result)
end