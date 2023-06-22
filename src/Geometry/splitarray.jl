######################
## Exports ###########
######################

export SplitArray

######################
## Split Array #######
###################### 

"""
Represents a static array with N dimensions whose size is
exactly 2 along each dimension. This can be used to store children in quadtrees, 
and faces of hypercubes. It also makes templating on
the number of dimensions easier than with a regular
SArray.
"""
struct SplitArray{N, T, L} <: StaticArray{NTuple{N, 2}, T, N}
    data::SArray{NTuple{N, 2}, T, N, L}
end

"""
Builds a SplitArray from a tuple of values.
"""
@generated function SplitArray(x::NTuple{L, T}) where {L, T}
    @assert log2(L) == round(log2(L)) "L must equal 2^N for integer N"
    N = Int(log2(L))
    :(SplitArray{$N, T, L}(x))
end

SplitArray(x::T...) where T = SplitArray((x...,))

Base.size(b::SplitArray) = size(b.data)
Base.length(b::SplitArray) = length(b.data)
Base.eachindex(b::SplitArray) = eachindex(b.data)
Base.getindex(b::SplitArray, I) = getindex(b.data, I)

"""
Highly specialized selectdim implementation for SplitArray (an array of size
2 along every dimension). Returns a SplitArray with N-1 dimensions. This is about
100 times faster than Julia's base selectdim().
"""
@generated function Base.selectdim(A::SplitArray{N, T}, d::Integer, i::Integer) where {N, T}
    quote
        x = 2^(d - 1)
        j = 1
        k = 1
        SplitArray($(Expr(:tuple, [quote
            z = j
            j += 1
            if k < x
                k += 1
            else
                j += x
                k = 1
            end
            A[z + (i - 1) * x]
        end for n in 1:(2^(N - 1))]...)))
    end
end