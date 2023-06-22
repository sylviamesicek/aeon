export STensor, AbstractTensor, order
export Covariant

#####################
## Tensor ###########
#####################

"""
Alias for an N-dimensional SArray
"""
const STensor{N, T, O, L} = SArray{NTuple{O, N}, T, O, L} where {N, T, O, L}

"""
An abstract tensor of order O
"""
abstract type AbstractTensor{N, T, O} end

order(::AbstractTensor{N, T, O}) where {N, T, O} = O

######################
## Covariant #########
######################

"""
A covariant tensor of a given rank.
"""
struct Covariant{N, T, O, L} <: AbstractTensor{N, T, O}
    inner::STensor{N, T, O, L}
end

transform(::Transform{N, T}, ::SVector{N, T}, tensor::Covariant{N, T, 0}) where {N, T} = tensor
transform(trans::Transform{N, T}, x::SVector{N, T}, tensor::Covariant{N, T, 1}) where {N, T} = Covariant(jacobian(trans, x) * tensor.inner)
function transform(trans::Transform{N, T}, x::SVector{N, T}, tensor::Covariant{N, T, 2}) where {N, T} 
    j = jacobian(trans, x)
    Covariant(transpose(j) * tensor.inner * j)
end