######################
## Exports ###########
######################

export SplitArray

######################
## Split Array #######
###################### 

"""
Represents a static array with N dimensions whose size is
exactly 2 along each dimension. This makes templating on
the number of dimensions easier than with a regular
SArray.
"""
struct SplitArray{N, T, L} <: StaticArray{NTuple{N, 2}, T, N}
    data::NTuple{L, T}
end

@generated function SplitArray(x::NTuple{L, T}) where {L, T}
    @assert log2(L) == round(log2(L)) "L must equal 2^N for integer N"
    N = Int(log2(L))
    :(SplitArray{$N, T, L}(x))
end

Base.getindex(b::SplitArray, i::Int) = b.data[i]

"""
Highly specialized selectdim implementation for SplitArray (an array of size
2 along every dimension). Returns a TwosArray with N-1 dimensions. See
`test/twosarray.jl` for exhaustive testing of this function. This is about
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
