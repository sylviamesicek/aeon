# Exports

export AnalyticField, AnalyticScalarField
export CombinedField, ScaledField

# Core

"""
A tensor field of rank `R` on a `N` dimensional flat manifold. Such fields
are covariant, as they often represent the result of derivative operations.
"""
abstract type AnalyticField{N, T, R} end

const AnalyticScalarField{N, T} = AnalyticField{N, T, 0}

"""
Returns the rank of a given tensor field.
"""
rank(::AnalyticField{N, T, R}) where {N, T, R} = R

(field::AnalyticField{N, T, R})(x::SVector{N, T}) where {N, T, R} = error("Application is unimplemented for $(typeof(field)) on $(typeof(x)).")

"""
A linear combination of tensor fields of the same rank and dimension.
"""
struct CombinedField{N, T, R, F} <: AnalyticField{N, T, R} 
    inner::F

    function CombinedField(x::AnalyticField{N, T, R}...) where {N, T, R}
        wrapped = (x...,)
        new{N, T, R, typeof(wrapped)}(wrapped)
    end
end

function (func::CombinedField{N, T})(x::SVector{N, T}) where {N, T}
    result = zero(T)
    for i in eachindex(func.inner)
        result += func.inner[i](x)
    end
    result
end

"""
Builds a linear combination of the given tensor fields.
"""
@inline Base.:(+)(first::CombinedField{N, T, R}, second::CombinedField{N, T, R}) where {N, T, R} = CombinedField(first.inner..., second.inner...)
@inline Base.:(+)(first::CombinedField{N, T, R}, second::AnalyticField{N, T, R}) where {N, T, R} = CombinedField(first.inner..., second)
@inline Base.:(+)(first::AnalyticField{N, T, R}, second::CombinedField{N, T, R}) where {N, T, R} = CombinedField(first, second.inner...)
@inline Base.:(+)(first::AnalyticField{N, T, R}, second::AnalyticField{N, T, R}) where {N, T, R} = CombinedField(first, second)

"""
Represents a tensor field which has been scaled by a certain constant.
"""
struct ScaledField{N, T, R, F} <: AnalyticField{N, T, R}
    scale::T
    inner::F

    ScaledField(scale::T, operator::AnalyticField{N, T, R}) where {N, T, R} = new{N, T, R, typeof(operator)}(scale, operator)
end

(field::ScaledField{N, T})(x::SVector{N, T}) where {N, T} = scale * field.inner(x)

"""
Builds a scaled tensor field.
"""
@inline Base.:(*)(scale::T1, field::AnalyticField{N, T2, R}) where {N, T1, T2, R} = ScaledField(convert(T2, scale), field)