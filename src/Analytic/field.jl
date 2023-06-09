###########################
## Exports ################
###########################

export AnalyticField, rank
export CombinedField

###########################
## Core ###################
###########################

"""
Represents an analytic rank-R tensor field defined across an N-dimensional manifold.
"""
abstract type AnalyticField{N, T, R} end

"""
    rank(field)

Returns the rank of the given tensor field
"""
rank(::AnalyticField{N, T, R}) where {N, T, R} = R

"""
    field(x)

Returns the value of the tensor field at the given position
"""
(field::AnalyticField{N, T})(x::SVector{N, T}) where {N, T} = error("Application is unimplemented for $(typeof(field)) on $(typeof(x))")

###########################
## Combine ################
###########################

"""
Represents a linear combination of functions (ie, the functions are summed when applied to a value).
"""
struct CombinedField{N, T, F, R} <: AnalyticField{N, T, R}
    functions::F

    function CombinedField(x::AnalyticField{N, T, R}...) where {N, T, R}
        wrapped = (x...,)
        new{N, T, typeof(wrapped), R}(wrapped)
    end
end

function (func::CombinedField{N, T})(x::SVector{N, T}) where {N, T}
    result = zero(T)
    for i in eachindex(func.functions)
        result += func.functions[i](x)
    end
    result
end

"""
Builds a combined function from two or more functions.
"""
@inline Base.:(+)(first::CombinedField{N, T, R}, second::CombinedField{N, T, R}) where {N, T, R} = CombinedField(first.functions..., second.functions...)
@inline Base.:(+)(first::AnalyticField{N, T, R}, second::CombinedField{N, T, R}) where {N, T, R} = CombinedField(first, second.functions...)
@inline Base.:(+)(first::CombinedField{N, T, R}, second::AnalyticField{N, T, R}) where {N, T, R} = CombinedField(first.functions..., second)
@inline Base.:(+)(first::AnalyticField{N, T, R}, second::AnalyticField{N, T, R}) where {N, T, R} = CombinedField(first, second)
