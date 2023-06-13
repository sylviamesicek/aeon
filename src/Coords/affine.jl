# Exports

export Translate, ScaleTransform, LinearTransform


#########################
## Translation ##########
#########################

"""
A simple translation.
"""
struct Translate{N, T} <: Transform{N, T}
    offset::SVector{N, T}
end

Translate(x::Tuple) = Translate(SVector(x))
Translate(x, y) = Translate(SVector(x, y))
Translate(x, y, z) = Translate(SVector(x, y, z))

(trans::Translate{N, T})(x::SVector{N, T}) where {N, T} = x + trans.offset
Base.inv(trans::Translate) = Translate(-trans.offset)
jacobian(::Translate{N, T}, x::SVector{N, T}) where {N, T} = I 

Base.:(∘)(trans1::Translate{N, T}, trans2::Translate{N, T}) where {N, T} = Translate(trans1.offset + trans2.offset)

Base.show(io::IO, trans::Translate) = print(io, "Translation$((trans.offset...,))")

########################
## Scale Transform #####
########################

struct ScaleTransform{N, T} <: Transform{N, T}
    scale::UniformScaling{T}

    ScaleTransform{N, T}(f::T) where {N, T} = new{N, T}(UniformScaling(f))
end

(trans::ScaleTransform{N, T})(x::SVector{N, T}) where {N, T} = trans * x 
Base.inv(trans::ScaleTransform{N, T}) where {N, T} = ScaleTransform{N, T}(inv(trans.scale))
jacobian(trans::ScaleTransform{N, T}, ::SVector{N, T}) where {N, T} = trans.scale

Base.:(∘)(t1::ScaleTransform{N, T}, t2::ScaleTransform{N, T}) where {N, T} = ScaleTransform{N, T}(t1.scale * t2.scale)

########################
## Linear Transform ####
########################

"""
    LinearTransform <: Transform
    LinearTransform(M)

A general linear transformation, constructed for any matrix-like object `M` using `LinearTransform(M)`.
"""
struct LinearTransform{N, T, M} <: Transform{N, T}
    linear::M

    LinearTransform{N, T}(linear::M) where {N, T, M} = new{N, T, M}(linear)
end

LinearTransform(linear::SMatrix{N, N, T, L}) where {N, T, L} = LinearTransform{N, T}(linear)

Base.show(io::IO, trans::LinearTransform) = print(io, "LinearTransform($(trans.linear))")

(trans::LinearTransform{N, T})(x::SVector{N, T}) where {N, T} = trans.linear * x
Base.inv(trans::LinearTransform{N, T}) where {N, T} = LinearTransform{N, T}(inv(trans.linear))
jacobian(trans::LinearTransform{N, T}, ::SVector{N, T}) where {N, T} = trans.linear

Base.:(∘)(t1::LinearTransform{N, T}, t2::LinearTransform{N, T}) where {N, T} = LinearTransform{N, T}(t1.linear * t2.linear)
