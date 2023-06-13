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

(func::AnalyticFunction{N, T})(::SVector{N, T}) where {N, T} = error("Application of $(typeof(func)) is unimplemented")

"""
Operation on an analytic function.
"""
abstract type AnalyticOperator{N, T, R} end

(operator::AnalyticOperator{N, T, R})(func::AnalyticFunction{N, T}) where {N, T, R} = error("Application of $(typeof(operator)) on $(typeof(func)) is unimplemented.")

####################
## Identity (Op) ###
####################

struct IdentityOperator{N, T} <: AnalyticOperator{N, T, T} end

(operator::IdentityOperator{N, T})(field::AnalyticFunction{N, T}) where {N, T} = field

#################
## Scale (Func) #
#################

"""
Represents a function which has been scaled by a certain constant.
"""
struct ScaledFunction{N, T, S, F} <: AnalyticFunction{N, T}
    scale::S
    inner::F

    ScaledFunction(scale, func::AnalyticFunction{N, T}) where {N, T} = new{N, T, typeof(scale), typeof(func)}(scale, func)
end

(func::ScaledFunction{N, T})(x::SVector{N, T}) where {N, T} = func.scale * func.inner(x)

"""
Builds a scaled function.
"""
@inline Base.:(*)(scale::T1, func::AnalyticFunction{N, T2}) where {N, T1, T2} = ScaledFunction(convert(T2, scale), func)

#################
## Scale (Op) ###
#################

struct ScaledOperator{N, T, R, S, O} <: AnalyticOperator{N, T, R}
    scale::S
    inner::O

    ScaledOperator(scale, operator::AnalyticOperator{N, T, R}) where {N, T, R} = new{N, T, R, typeof(scale), typeof(operator)}(scale, operator)
end

(operator::ScaledOperator{N, T})(field::AnalyticFunction{N, T}) where {N, T,} = ScaledFunction(operator.scale, operator(field))

"""
Builds a scaled operator
"""
@inline Base.:(*)(scale::T1, oper::AnalyticOperator{N, T2}) where {N, T1, T2} = ScaledOperator(convert(T2, scale), oper)

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
    inner::O

    function CombinedOperator(x::AnalyticOperator{N, T, R}...) where {N, T, R}
        wrapped = (x...,)
        new{N, T, R, typeof(wrapped)}(wrapped)
    end
end

# TODO Speed this up with generated function
(operator::CombinedOperator{N, T})(field::AnalyticFunction{N, T}) where {N, T} = CombinedFunction(map(op -> op(field), operator.inner)...)

@inline Base.:(+)(first::CombinedOperator{N, T, R}, second::CombinedOperator{N, T, R}) where {N, T, R} = CombinedOperator(first.inner..., second.inner...)
@inline Base.:(+)(first::CombinedOperator{N, T, R}, second::AnalyticOperator{N, T, R}) where {N, T, R} = CombinedOperator(first.inner..., second)
@inline Base.:(+)(first::AnalyticOperator{N, T, R}, second::CombinedOperator{N, T, R}) where {N, T, R} = CombinedOperator(first, second.inner...)
@inline Base.:(+)(first::AnalyticOperator{N, T, R}, second::AnalyticOperator{N, T, R}) where {N, T, R} = CombinedOperator(first, second)

##########################
## Transformed (Func) ####
##########################

struct TransformedFunction{N, T, Tr, F} <: AnalyticFunction{N, T} 
    transform::Tr
    inner::F

    TransformedFunction{N, T}(trans::Tr, func::F) where {N, T, F <: AnalyticFunction{N, T}, Tr <: Transform{N, T}} = new{N, T, Tr, F}(trans, func)
end

(func::TransformedFunction{N, T})(x::SVector{N, T}) where {N, T} = func.inner(func.transform(x))

##########################
## Transformed Operator ##
##########################

struct TransformedOperator{N, T, R, Tr, O} <: AnalyticOperator{N, T, R}
    transform::Tr
    inner::O

    TransformedOperator{N, T, R}(trans::Tr, oper::O) where {N, T, R, Tr <: Transform{N, T}, O <: AnalyticOperator{N, T, R}} = new{N, T, R, Tr, O}(trans, oper)
end

(operator::TransformedOperator{N, T})(func::AnalyticFunction{N, T}) where {N, T} = operator.inner(TransformedFunction{N, T}(operator.transform, func))