###########################
## Exports ################
###########################

export CombineOperator, CombinedFunction, combine

#########################
## Combination ##########
#########################

"""
Represents a linear combination of operators.
"""
struct CombineOperator{O} <: AnalyticOperator
    operators::O
end

function (oper::CombineOperator)(func::AnalyticFunction)::CombinedFunction
    CombinedFunction((map(op -> op(func), oper.operators)...,))
end

@inline combine(x::AnalyticOperator...) = CombineOperator((x...,))

"""
Builds a combine operator from two or more analytic operators
"""
@inline Base.:(+)(first::S, second::T) where {S<:AnalyticOperator, T<:AnalyticOperator} = combine(first, second)
@inline Base.:(+)(first::CombineOperator{S}, second::CombineOperator{T}) where {S, T} =  combine(first.operators..., second.operators...)
@inline Base.:(+)(first::CombineOperator{S}, second::T) where {S, T<:AnalyticOperator} = combine(first.operators..., second)
@inline Base.:(+)(first::S, second::CombineOperator{T}) where {S<:AnalyticOperator, T} = combine(first, second.operators...)

"""
Represents a linear combination of functions (ie, the functions are summed when applied to a value).
"""
struct CombinedFunction{O} <: AnalyticFunction
    functions::O
end

function (func::CombinedFunction{O})(x::AbstractVector{F}) where {O, F}
    result = zero(F)
    for i in eachindex(O.parameters)
        result += func.functions[i](x)
    end
    result
end

@inline combine(x::AnalyticFunction...) = CombinedFunction((x...,))

"""
Builds a combined function from two or more functions
"""
@inline Base.:(+)(first::S, second::T) where {S<:AnalyticFunction, T<:AnalyticFunction} = combine(first, second)
@inline Base.:(+)(first::CombinedFunction{S}, second::CombinedFunction{T}) where {S, T} =  combine(first.functions..., second.functions...)
@inline Base.:(+)(first::CombinedFunction{S}, second::T) where {S, T<:AnalyticFunction} = combine(first.functions..., second)
@inline Base.:(+)(first::S, second::CombinedFunction{T}) where {S<:AnalyticFunction, T} = combine(first, second.functions...)