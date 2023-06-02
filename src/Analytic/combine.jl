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

@inline combine(x...) = CombineOperator((x...,))

# """
# Builds a Combine Operator from two or more analytic operators
# """
# Base.:(+)(first::S, second::T) where {S<:AnalyticOperator, T<:AnalyticOperator} = CombineOperator{Tuple{S, T}}((first, second))
# Base.:(+)(first::CombineOperator{S}, second::CombineOperator{T}) where {S, T} = CombineOperator{Tuple{S.parameters..., T.parameters...}}((first.operators..., second.operators...))
# Base.:(+)(first::CombineOperator{S}, second::T) where {S, T<:AnalyticOperator} = CombineOperator{Tuple{S.parameters..., T}}((first.operators..., second))
# Base.:(+)(first::S, second::CombineOperator{T}) where {S<:AnalyticOperator, T} = CombineOperator{Tuple{S, T.parameters...}}((first, second.operators...))

"""
Builds a Combine Operator from two or more analytic operators
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

function (func::CombinedFunction{O})(x::AbstractVector{F})::F where {O, F}
    result = zero(F)
    for i in eachindex(O.parameters)
        result += func.functions[i](x)
    end
    result
end