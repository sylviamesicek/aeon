# Exports

export Covariant

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