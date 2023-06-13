# Exports

# Indices

"""
An N-dimensional tensor with of rank R defined over a field `T`
"""
abstract type AbstractTensor{N, R, T} end

struct Tensor{N, R, T, L} 
    data::NTuple{L, T}

    function Tensor{N, R, T, L}(data::NTuple{L, Number}) where {N, R, T, L}
        check_tensor_parameters(N, R, T, L)
        new{N, R, T, L}(convert_ntuple(T, data))
    end
end

@generated function check_tensor_parameters(::Type{N}, ::Type{R}, ::Type{T}, ::Type{L}) where {N, R, T, L}
    if N^R ≠ L
        return :(throw(ArgumentError("Length of tuple data must be $(N^R) got $(L)")))
    end
end

# Cast any Tuple to an TupleN{T}
@generated function convert_ntuple(::Type{T}, d::NTuple{N,Any}) where {N,T}
    exprs = ntuple(i -> :(convert(T, d[$i])), Val(N))
    return quote
        Base.@_inline_meta
        $(Expr(:tuple, exprs...))
    end
end

@generated function tensorsize(::Val{N}, ::Val{R}) where {N, R}
    exprs = ntuple(_ -> N, R)
    :($(exprs...,))
end