# Exports

export AFunction, AFunctional
export AIdentity, AScaled, ACombined, ATransformed

################
## Core ########
################

"""
Represents an N dimensional analytical function of type T.
"""
abstract type AFunction{N, T} end

(func::AFunction{N, T})(::SVector{N, T}) where {N, T} = error("Application of $(typeof(func)) is unimplemented")

"""
Represents an analytic functional over N dimensional functions.
"""
abstract type AFunctional{N, T, R} end

(operator::AFunctional{N, T, R})(func::AFunction{N, T}, position::SVector{N, T}) where {N, T, R} = error("Application of $(typeof(operator)) on $(typeof(func)) is unimplemented.")

####################
## Identity ########
####################

struct AIdentity{N, T} <: AFunctional{N, T, T} end

(operator::AIdentity{N, T})(func::AFunction{N, T}, position::SVector{N, T}) where {N, T} = func(position)

#################
## Scale ########
#################

"""
The composed operations of applying an operator to a function and then scaling it.
"""
struct AScaled{N, T, R, O} <: AFunctional{N, T, R}
    scale::T
    inner::O

    AScaled(scale::T, operator::AFunctional{N, T, R}) where {N, T, R} = new{N, T, R, typeof(operator)}(scale, operator)
end

(operator::AScaled{N, T})(func::AFunction{N, T}, position::SVector{N, T}) where {N, T} = operator.scale * operator.inner(func, position)

"""
Builds a scaled operator
"""
@inline Base.:(*)(scale::S, oper::AFunctional{N, T}) where {N, S, T} = AScaled(convert(T, scale), oper)

##################
## Combined ######
##################

struct ACombined{N, T, R, O} <: AFunctional{N, T, R}
    inner::O

    function ACombined(x::AFunctional{N, T, R}...) where {N, T, R}
        wrapped = (x...,)
        new{N, T, R, typeof(wrapped)}(wrapped)
    end
end

# TODO Speed this up with generated function
function (operator::ACombined{N, T, R})(func::AFunction{N, T}, position::SVector{N, T}) where {N, T, R}
    result = zero(R)
    for i in eachindex(operator.inner)
        result += operator.inner[i](func, position)
    end
    result
end

@inline Base.:(+)(first::ACombined{N, T, R}, second::ACombined{N, T, R}) where {N, T, R} = ACombined(first.inner..., second.inner...)
@inline Base.:(+)(first::ACombined{N, T, R}, second::AFunctional{N, T, R}) where {N, T, R} = ACombined(first.inner..., second)
@inline Base.:(+)(first::AFunctional{N, T, R}, second::ACombined{N, T, R}) where {N, T, R} = ACombined(first, second.inner...)
@inline Base.:(+)(first::AFunctional{N, T, R}, second::AFunctional{N, T, R}) where {N, T, R} = ACombined(first, second)

##########################
## Transformed ###########
##########################

struct ATransformed{N, T, R, S, O} <: AFunctional{N, T, R} 
    trans::S
    inner::O

    ATransformed(trans::Transform{N, T}, operator::AFunctional{N, T, R}) where {N, T, R} = new{N, T, R, typeof(trans), typeof(O)}(trans, operator)
end

function (operator::ATransformed{N, T, R})(func::AFunction{N, T}, position::SVector{N, T}) where {N, T, R}
    tpos = operator.trans(position)
    result = operator.inner(func, tpos)
    transform(operator.trans, position, result)
end