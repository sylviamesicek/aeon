# Exports

export AnalyticOperator, AnalyticValue, AnalyticDerivative, AnalyticCurvature
export CombinedOperator, ScaledOperator
export derivative, curvature

# Operator

"""
Represents a tensorial operator on an analytic tensor field. An operator acts on a scalar field, and returns a tensor field of rank R.
"""
abstract type AnalyticOperator{N, T, R} end

"""
An analytic operator acts on a scalar field to produce a covariant tensor field. This is most commonly some sort of derivative operation.
"""
(operator::AnalyticOperator{N, T})(field::AnalyticScalarField{N, T}) where {N, T}  = error("Application is unimplemented for $(typeof(operator)) on $(typeof(field))")

# Value

"""
Represents the value operation.
"""
struct AnalyticValue{N, T} <: AnalyticOperator{N, T, 0} end

(::AnalyticValue{N, T})(field::AnalyticScalarField{N, T}) where {N, T} = field

# Derivative

struct AnalyticDerivative{N, T} <: AnalyticOperator{N, T, 1} end

derivative(::AnalyticValue{N, T}) where {N, T} = AnalyticDerivative{N, T}()

# Curvature

struct AnalyticCurvature{N, T} <: AnalyticOperator{N, T, 2} end

curvature(::AnalyticValue{N, T}) where {N, T} = AnalyticCurvature{N, T}()

# Combine

struct CombinedOperator{N, T, R, O} <: AnalyticOperator{N, T, R}
    inner::O

    function CombinedOperator(x::AnalyticOperator{N, T, R}...) where {N, T, R}
        wrapped = (x...,)
        new{N, T, R, typeof(wrapped)}(wrapped)
    end
end

function (oper::CombinedOperator{N, T})(field::AnalyticScalarField{N, T}) where {N, T}
    CombinedField(map(op -> op(field), oper.inner)...)
end

@inline Base.:(+)(first::CombinedOperator{N, T, R}, second::CombinedOperator{N, T, R}) where {N, T, R} = CombinedOperator(first.inner..., second.inner...)
@inline Base.:(+)(first::CombinedOperator{N, T, R}, second::AnalyticOperator{N, T, R}) where {N, T, R} = CombinedOperator(first.inner..., second)
@inline Base.:(+)(first::AnalyticOperator{N, T, R}, second::CombinedOperator{N, T, R}) where {N, T, R} = CombinedOperator(first, second.inner...)
@inline Base.:(+)(first::AnalyticOperator{N, T, R}, second::AnalyticOperator{N, T, R}) where {N, T, R} = CombinedOperator(first, second)

# Scale

struct ScaledOperator{N, T, R, O} <: AnalyticOperator{N, T, R}
    scale::T
    inner::O

    ScaledField(scale::T, operator::AnalyticOperator{N, T, R}) where {N, T, R} = new{N, T, R, typeof(operator)}(scale, operator)
end

(operator::ScaledOperator{N, T})(field::AnalyticScalarField{N, T}) where {N, T} = ScaledField(operator.scale, operator.inner(field))

@inline Base.:(*)(scale::T1, field::AnalyticOperator{N, T2, R}) where {N, T1, T2, R} = ScaledOperator(convert(T2, scale), field)
