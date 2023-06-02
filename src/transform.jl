########################
## Exports #############
########################

export Transform, IdentityTransform, ComposedTransform, ∘, jacobian
export Translate, ScaleTransform, LinearTransform

########################
## Transform ###########
########################

"""
The `Transform` supertype defines a simple Transform which may be applied to coordinate vectors.
"""
abstract type Transform end

"""
    trans(position)

Applies a transformation to a given point.
"""
(trans::Transform)(x) = error("Transformation for $(typeof(trans)) has not been defined.")

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
jacobian(trans::Transform, x) = error("Differential matrix of transform $trans with input $x has not been defined.")

########################
## Identity ############
########################

"""
An identity transform which acts similarly to the identity function.
"""
struct IdentityTransform <: Transform; end

@inline (::IdentityTransform)(x) = x
@inline Base.inv(trans::IdentityTransform) = trans
@inline jacobian(::IdentityTransform) = I

#######################
## Composition ########
#######################

"""
Defines a composed transformation which first applies t2 and then t1.
"""
struct ComposedTransform{T1 <: Transform, T2 <: Transform} <: Transform 
    t1::T1
    t2::T2
end

"""
    trans1 ∘ trans2

Take two transformations and create a new transformation which is equivilent to
successively applying `trans2` to the coordinate and then `trans`.
"""
∘(trans1::Transform, trans2::Transform) = ComposedTransform(trans1, trans2)
∘(trans::IdentityTransform, ::IdentityTransform) = trans
∘(::IdentityTransform, trans::Transform) = trans
∘(trans::Transform, ::IdentityTransform) = trans

(trans::ComposedTransform)(x) = trans.t1(trans.t2(x))

Base.inv(trans::ComposedTransform) = inv(trans.t2) ∘ inv(trans.t1)

function jacobian(trans::ComposedTransform, x)
    x2 = trans.t2(x)
    m1 = jacobian(trans.t1, x2)
    m2 = jacobian(trans.t2, x)
    m1 * m2
end

Base.show(io::IO, trans::ComposedTransform) = print(io, "($(trans.t1) ∘ $(trans.t2))")

#########################
## Translation ##########
#########################

"""
A simple translation.
"""
struct Translate{V} <: Transform
    offset::V
end

Translate(x::Tuple) = Translate(SVector(x))
Translate(x, y) = Translate(SVector(x, y))
Translate(x, y, z) = Translate(SVector(x, y, z))

(trans::Translate{V})(x) where {V} = x + trans.offset
Base.inv(trans::Translate) = Translate(-trans.offset)
jacobian(trans::Translate, x) = I 

∘(trans1::Translate, trans2::Translate) = Translate(trans1.offset + trans2.offset)

Base.show(io::IO, trans::Translate) = print(io, "Translation$((trans.offset...,))")

########################
## Scale Transform #####
########################

struct ScaleTransform{F<:Number} <: Transform
    scale::UniformScaling{F}

    ScaleTransform(f) = new{F}(UniformScaling(f))
end

(trans::ScaleTransform{F})(x) where {F<:Number} = trans * x 
(trans::ScaleTransform{F})(x::Tuple) where {F} = trans(SVector(x))
Base.inv(trans::ScaleTransform) = ScaleTransform(inv(trans.scale))
jacobian(trans::ScaleTransform, x) = trans.scale

∘(t1::ScaleTransform, t2::ScaleTransform) = ScaleTransform(t1.scale * t2.scale)

########################
## Linear Transform ####
########################

"""
    LinearTransform <: Transform
    LinearTransform(M)

A general linear transformation, constructed for any matrix-like object `M` using `LinearTransform(M)`.
"""
struct LinearTransform{M} <: Transform
    linear::M
end

Base.show(io::IO, trans::LinearTransform) = print(io, "LinearMap($(trans.linear))")

(trans::LinearTransform{M})(x) where {M} = trans.linear * x
(trans::LinearTransform{M})(x::Tuple) where {M} = trans(SVector(x))
Base.inv(trans::LinearTransform) = LinearTransform(inv(trans.linear))
jacobian(trans::LinearTransform, x) = trans.linear

∘(t1::LinearTransform, t2::LinearTransform) = LinearTransform(t1.linear * t2.linear)