###################
## Export #########
###################

export AnalyticOperator, rank
export ValueOperator, GradientOperator, DerivativeOperator, HessianOperator, CurvatureOperator, LaplacianOperator
export ∇, ∇², Δ
export ScaledOperator, ScaledOperator, ScaledField
export CombineOperator

###################
## Operator #######
###################

"""
An analytic operation on a tensor field.
"""
abstract type AnalyticOperator{N, T} end

(operator::AnalyticOperator{N, T})(field::AnalyticField{N, T}) where {N, T} = error("Application is unimplemented for $(typeof(operator)) on $(typeof(field))")

###################
## Value ##########
###################

struct ValueOperator{N, T} <: AnalyticOperator{N, T} end

(::ValueOperator)(field::AnalyticField) = field

###################
## Gradient #######
###################

struct GradientOperator{N, T} <: AnalyticOperator{N, T} end

(::GradientOperator)(field::AnalyticField) = error("Gradient of $(typeof(field)) is undefined")

∇(::ValueOperator{N, T}) where {N, T} = GradientOperator{N, T}()

###################
## Derivative #####
###################

struct DerivativeOperator{N, T} <: AnalyticOperator{N, T}
    normal::SVector{N, T}

    DerivativeOperator(normal::SVector{N, T}) where {N, T} = new{N, T}(normal)
end

function (operator::DerivativeOperator{N, T})(field::AnalyticField{N, T}) where {N, T}
    dot(operator.normal, GradientOperator{N, T}()(field))
end

LinearAlgebra.:(⋅)(normal::SVector{N, T}, ::GradientOperator{N, T}) where {N, T} = DerivativeOperator(normal)

##################
## Hessian #######
##################

struct HessianOperator{N, T} <: AnalyticOperator{N, T} end

(::HessianOperator)(field::AnalyticField) = error("Hessian of $(typeof(field)) is undefined")

∇²(::ValueOperator{N, T}) where {N, T} = HessianOperator{N, T}()

##################
## Curvature #####
##################

struct CurvatureOperator{N, T, S} <: AnalyticOperator{N, T}
    direction::SMatrix{N, N, T, S}

    CurvatureOperator(direction::SMatrix{N, N, T, S}) where {N, T, S} = new{N, T, S}(direction)
end

function (operator::CurvatureOperator{N, T})(field::AnalyticField{N, T}) where {N, T}
    dot(operator.direction, HessianOperator{N, T}()(field))
end

LinearAlgebra.:(⋅)(direction::SMatrix{N, N, T}, ::HessianOperator{N, T}) where {N, T} = CurvatureOperator(direction)

##################
## Laplacian #####
##################

struct LaplacianOperator{N, T} <: AnalyticOperator{N, T} end

function (operator::LaplacianOperator{N, T})(field::AnalyticField{N, T}) where {N, T}
    dot(I, HessianOperator{N, T}()(field))
end

Δ(::ValueOperator{N, T}) where {N, T} = LaplacianOperator{N, T}()

##################
## Compose #######
##################

# struct ComposedOperator{N, T, O1, O2} <: AnalyticOperator{N, T}
#     o1::O1
#     o2::O2

#     ComposedOperator(o1::AnalyticOperator{N, T}, o2::AnalyticOperator{N, T}) where{N, T} = new{N, T, typeof(o1), typeof(o2)}(o1, o2)
# end

# (operator::ComposedOperator{N, T})(field::AnalyticField{N, T}) = operator.o1(operator.o2(field))

##################
## Scale #########
##################

struct ScaleOperator{N, T} <: AnalyticOperator{N, T}
    scale::T

    ScaleOperator{N}(scale::T) where {N, T} = new{N, T}(scale)
end

struct ScaledOperator{N, T, O} <: AnalyticOperator{N, T}
    scale::T
    operator::O

    ScaledOperator(scale::T, operator::AnalyticOperator{N, T}) where {N, T} = new{N, T, typeof(operator)}(scale, operator)
end

struct ScaledField{N, T, F, R} <: AnalyticField{N, T, R}
    scale::T
    field::F

    ScaledField(scale::T, field::AnalyticField{N, T, R}) where {N, T, R} = new{N, T, typeof(field), R}(scale, field)
end

rank(field::ScaledField) = rank(field.field)

(operator::ScaleOperator{N, T})(field::AnalyticField{N, T}) where {N, T} = ScaledField(operator.scale, field)

Base.:(*)(scale::T2, field::AnalyticField{N, T}) where {N, T, T2} = ScaledField(convert(T, scale), field)
Base.:(*)(scale::T2, operator::AnalyticOperator{N, T}) where {N, T, T2} = ScaledOperator(convert(T, scale), operator)

#####################
## Combine ##########
#####################

"""
Represents a linear combination of operators.
"""
struct CombineOperator{N, T, O} <: AnalyticOperator{N, T}
    operators::O

    function CombineOperator(x::AnalyticOperator{N, T}...) where {N, T}
        wrapped = (x...,)
        new{N, T, typeof(wrapped)}(wrapped)
    end
end

function (oper::CombineOperator{N, T})(field::AnalyticField{N, T}) where {N, T}
    CombinedField(map(op -> op(field), oper.operators)...,)
end

"""
Builds a combine operator from two or more analytic operators
"""
@inline Base.:(+)(first::CombineOperator{N, T}, second::CombineOperator{N, T}) where {N, T} = CombineOperator(first.operators..., second.operators...)
@inline Base.:(+)(first::AnalyticOperator{N, T}, second::CombineOperator{N, T}) where {N, T} = CombineOperator(first, second.operators...)
@inline Base.:(+)(first::CombineOperator{N, T}, second::AnalyticOperator{N, T}) where {N, T} = CombineOperator(first.operators..., second)
@inline Base.:(+)(first::AnalyticOperator{N, T}, second::AnalyticOperator{N, T}) where {N, T} = CombineOperator(first, second)

########################
## Implicit ############
########################


