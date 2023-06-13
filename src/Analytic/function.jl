# Exports

export AnalyticFunction, AnalyticOperator
export ScaledFunction, ScaledOperator
export CombinedFunction, CombinedOperator

################
## Core ########
################

"""
Represents an N dimensional analytical function of type T.
"""
abstract type AnalyticFunction{N, T} end

(func::AnalyticFunction{N, T})(::SVector{N, T})::T where {N, T} = error("Application of $(typeof(func)) is unimplemented")

"""
Operation on an analytic function.
"""
abstract type AnalyticOperator{N, T, R} end

(operator::AnalyticOperator{N, T, R})(func::AnalyticFunction{N, T})::AnalyticFunction{N, R} where {N, T, R} = error("Application of $(typeof(operator)) on $(typeof(func)) is unimplemented.")

#################
## Scale (Func) #
#################

"""
Represents a function which has been scaled by a certain constant.
"""
struct ScaledFunction{N, T, F} <: AnalyticFunction{N, T}
    scale::T
    inner::F

    ScaledFunction(scale::T, func::AnalyticFunction{N, T}) where {N, T} = new{N, T, typeof(func)}(scale, func)
end

(field::ScaledField{N, T})(x::SVector{N, T})::T where {N, T} = field.scale * field.inner(x)

#################
## Scale (Op) ###
#################

"""
Builds a scaled tensor field.
"""
@inline Base.:(*)(scale::T1, field::AnalyticField{N, T2}) where {N, T1, T2} = ScaledFunction(convert(T2, scale), field)

struct ScaledOperator{N, T, R, O} <: AnalyticOperator{N, T, R}
    scale::T
    inner::O

    ScaledOperator(scale::T, operator::AnalyticOperator{N, T, R}) where {N, T, R} = new{N, T, R, typeof(operator)}(scale, operator)
end

(operator::ScaledOperator{N, T, R})(field::AnalyticFunction{N, T}) = operator.inner(ScaledFunction(operator.scale, field))

####################
## Combined (Func) #
####################

struct CombinedFunction{N, T, F} <: AnalyticFunction{N, T}
    inner::Vector{F}

    function CombinedFunction(x::AnalyticFunction{N, T}...) where {N, T}
        wrapped = (x...,)
        new{N, T, typeof(wrapped)}(wrapped)
    end
end

function (func::CombinedFunction{N, T})(x::SVector{N, T}) where {N, T}
    result = zero(T)
    for i in eachindex(func.inner)
        result += func.inner[i](x)
    end
    result
end

"""
Builds a linear combination of the given functions.
"""
@inline Base.:(+)(first::CombinedFunction{N, T}, second::CombinedFunction{N, T}) where {N, T} = CombinedFunction(first.inner..., second.inner...)
@inline Base.:(+)(first::CombinedFunction{N, T}, second::AnalyticFunction{N, T}) where {N, T} = CombinedFunction(first.inner..., second)
@inline Base.:(+)(first::AnalyticFunction{N, T}, second::CombinedFunction{N, T}) where {N, T} = CombinedFunction(first, second.inner...)
@inline Base.:(+)(first::AnalyticFunction{N, T}, second::AnalyticFunction{N, T}) where {N, T} = CombinedFunction(first, second)

##################
## Combined (Op) #
##################

struct CombinedOperator{N, T, R, O} <: AnalyticOperator{N, T, R}
    inner::Vector{O}

    function CombinedOperator(x::AnalyticOperator{N, T, R}...) where {N, T, R}
        wrapped = (x...,)
        new{N, T, R, typeof(wrapped)}(wrapped)
    end
end

(operator::CombinedOperator{N, T, R})(field::AnalyticFunction{N, T}) = operator.inner(CombinedFunction(map(op -> op(field), operator.inner)))

@inline Base.:(+)(first::CombinedOperator{N, T, R}, second::CombinedOperator{N, T, R}) where {N, T, R} = CombinedOperator(first.inner..., second.inner...)
@inline Base.:(+)(first::CombinedOperator{N, T, R}, second::AnalyticOperator{N, T, R}) where {N, T, R} = CombinedOperator(first.inner..., second)
@inline Base.:(+)(first::AnalyticOperator{N, T, R}, second::CombinedOperator{N, T, R}) where {N, T, R} = CombinedOperator(first, second.inner...)
@inline Base.:(+)(first::AnalyticOperator{N, T, R}, second::AnalyticOperator{N, T, R}) where {N, T, R} = CombinedOperator(first, second)

