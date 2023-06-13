# Exports

export Covariant, order

######################
## Covariant #########
######################

"""
A covariant tensor of a given rank.
"""
struct Covariant{N, T, O, L}
    inner::SArray{NTuple{O, N}, T, O, L}

    Covariant(tensor::SArray{NTuple{O, N}, T, O, L}) where {N, T, O, L} = new{N, T, O, L}(tensor)
end



order(::Covariant{N, T, O}) where {N, T, O} = O

transform(::Transform{N, T}, ::SVector{N, T}, tensor::Covariant{N, T, 0}) where {N, T} = tensor
transform(trans::Transform{N, T}, x::SVector{N, T}, tensor::Covariant{N, T, 1}) where {N, T} = Covariant(jacobian(trans, x) * tensor.inner)
function transform(trans::Transform{N, T}, x::SVector{N, T}, tensor::Covariant{N, T, 2}) where {N, T} 
    j = jacobian(trans, x)
    transpose(j) * tensor.inner * j
end