########################
## Exports #############
########################

export Transform, IdentityTransform, ComposedTransform
export jacobian


########################
## Transform ###########
########################

"""
The `Transform` supertype defines a simple Transform which may be applied to N-dimensional coordinate vectors.
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

# """
#     transform(trans, x, geom)

# Transforms a given geometric object which may be dependent on the coordinate system.
# """
# transform(trans::Transform{N, T}, ::SVector{N, T}, geom) where {N, T} = error("Transformation of object $(typeof(obj)) by $(typeof(trans)) is unimplemented.")
# transform(::Transform{N, T}, ::SVector{N, T}, geom::T) where {N, T} = geom

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