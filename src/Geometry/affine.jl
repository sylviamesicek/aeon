# Exports

export Translate, UniformScaleTransform, LinearTransform, ScaleTransform

#########################
## Translation ##########
#########################

"""
A simple translation.
"""
struct Translate{N, T} <: Transform{N, T}
    offset::SVector{N, T}
end

Translate(x) = Translate(SVector(x))
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

# Uniform

"""
A uniform scaling of a coordinate system in each direction.
"""
struct UniformScaleTransform{N, T} <: Transform{N, T}
    scale::UniformScaling{T}

    UniformScaleTransform{N, T}(f::T) where {N, T} = new{N, T}(UniformScaling(f))
end

(trans::UniformScaleTransform{N, T})(x::SVector{N, T}) where {N, T} = trans.scale * x 
Base.inv(trans::UniformScaleTransform{N, T}) where {N, T} = UniformScaleTransform{N, T}(inv(trans.scale))
jacobian(trans::UniformScaleTransform{N, T}, ::SVector{N, T}) where {N, T} = trans.scale

Base.:(∘)(t1::UniformScaleTransform{N, T}, t2::UniformScaleTransform{N, T}) where {N, T} = UniformScaleTransform{N, T}(t1.scale * t2.scale)

# Non-uniform

"""
A transform which arbitrarily scales each axis by some value.
"""
struct ScaleTransform{N, T} <: Transform{N, T}
    scales::Diagonal{T, SVector{N, T}}

    ScaleTransform(scales::SVector{N, T}) where {N, T} = new{N, T}(Diagonal(scales))
    ScaleTransform(scales::Vararg{T, N}) where {N, T} = ScaleTrasform(SVector(scales)) 
end

Base.show(io::IO, scale::ScaleTransform) = print(io, "Scale($(scale.scales.diag))")

(trans::ScaleTransform{N, T})(x::SVector{N, T}) where {N, T} = trans.scales * x 
Base.inv(trans::ScaleTransform{N, T}) where {N, T} = ScaleTransform{N, T}(inv(trans.scales))
jacobian(trans::ScaleTransform{N, T}, ::SVector{N, T}) where {N, T} = trans.scales

Base.:(∘)(t1::ScaleTransform{N, T}, t2::ScaleTransform{N, T}) where {N, T} = UniformScaleTransform{N, T}(t1.scales * t2.scales)

########################
## Linear Transform ####
########################

"""
    LinearTransform <: Transform
    LinearTransform(M)

A general linear transformation.
"""
struct LinearTransform{N, T, L} <: Transform{N, T}
    linear::SMatrix{N, N, T, L}
end

Base.show(io::IO, trans::LinearTransform) = print(io, "LinearTransform($(trans.linear))")

(trans::LinearTransform{N, T})(x::SVector{N, T}) where {N, T} = trans.linear * x
Base.inv(trans::LinearTransform{N, T}) where {N, T} = LinearTransform{N, T}(inv(trans.linear))
jacobian(trans::LinearTransform{N, T}, ::SVector{N, T}) where {N, T} = trans.linear

Base.:(∘)(t1::LinearTransform{N, T}, t2::LinearTransform{N, T}) where {N, T} = LinearTransform{N, T}(t1.linear * t2.linear)
