########################
## Exports #############
########################

export Transform, IdentityTransform, ComposedTransform, jacobian, transform
export Translate, ScaleTransform, LinearTransform

export Covariant, order

########################
## Transform ###########
########################

"""
The `Transform` supertype defines a simple Transform which may be applied to coordinate vectors.
"""
abstract type Transform{N, T} end

"""
    trans(position)

Applies a transformation to a given point. In order to avoid confusion, this may only be applied to coordinate vectors.
For transformation of more general objects, see the `transform` function.
"""
(trans::Transform{N, T})(::SVector{N, T}) where {N, T} = error("Transformation for $(typeof(trans)) has not been defined.")

"""
    inv(trans::Transform)

Computes the inverse of a coordinate transformation. This produces a transform
which undoes the effects of the original transform.
"""
Base.inv(trans::Transform) = error("Inverse transformation for $(typeof(trans)) has not been defined.")

"""
    jacobian(trans::Transform, x)

Returns a matrix describing how differential on the parameters of `x` flow through to
the output of the transformation `trans`
"""
jacobian(trans::Transform{N, T}, x::SVector{N, T}) where {N, T} = error("Differential matrix of transform $(typeof(trans)) with input $(typeof(x)) has not been defined.")

"""
    transform(trans, x, geom)

Transforms a given geometric object which may be dependent on the coordinate system.
"""
transform(trans::Transform{N, T}, ::SVector{N, T}, geom) where {N, T} = error("Transformation of object $(typeof(obj)) by $(typeof(trans)) is unimplemented.")
transform(::Transform{N, T}, ::SVector{N, T}, geom::T) where {N, T} = geom

########################
## Identity ############
########################

"""
An identity transform which acts similarly to the identity function.
"""
struct IdentityTransform{N, T} <: Transform{N, T} end

@inline (::IdentityTransform{N, T})(x::SVector{N, T}) where {N, T}= x
@inline Base.inv(trans::IdentityTransform{N, T}) where {N, T} = trans
@inline jacobian(::IdentityTransform{N, T}) where {N, T} = I

#######################
## Composition ########
#######################

"""
Defines a composed transformation which first applies t2 and then t1.
"""
struct ComposedTransform{N, T, T1, T2} <: Transform{N, T}
    t1::T1
    t2::T2

    ComposedTransform(t1::Transform{N, T}, t2::Transform{N, T}) where {N, T} = new{N, T, typeof(t1), typeof(t2)}(t1, t2)
end

"""
    trans1 ∘ trans2

Take two transformations and create a new transformation which is equivilent to
successively applying `trans2` to the coordinate and then `trans`.
"""
Base.:(∘)(trans1::Transform{N, T}, trans2::Transform{N, T}) where {N, T} = ComposedTransform(trans1, trans2)
Base.:(∘)(trans::IdentityTransform{N, T}, ::IdentityTransform{N, T}) where {N, T} = trans
Base.:(∘)(::IdentityTransform{N, T}, trans::Transform{N, T}) where {N, T} = trans
Base.:(∘)(trans::Transform{N, T}, ::IdentityTransform{N, T}) where {N, T} = trans

(trans::ComposedTransform{N, T})(x::SVector{N, T}) where {N, T} = trans.t1(trans.t2(x))

Base.inv(trans::ComposedTransform) = inv(trans.t2) ∘ inv(trans.t1)

function jacobian(trans::ComposedTransform{N, T}, x::SVector{N, T}) where {N, T}
    x2 = trans.t2(x)
    m1 = jacobian(trans.t1, x2)
    m2 = jacobian(trans.t2, x)
    m1 * m2
end

Base.show(io::IO, trans::ComposedTransform{N, T}) where {N, T} = print(io, "($(trans.t1) ∘ $(trans.t2))")

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