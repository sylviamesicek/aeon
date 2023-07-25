#####################
## Operator #########
#####################

export AbstractOperator, operator_apply!, operator_diagonal!

abstract type AbstractOperator{T} end

operator_apply!(y::AbstractVector{T}, oper::AbstractOperator{T}, x::AbstractVector{T}, maxlevel::Int) where T = error("Unimplemented")
operator_diagonal!(y::AbstractVector{T}, oper::AbstractOperator{T}, maxlevel::Int) where T = error("Unimplemented")
operator_restrict!(y::AbstractVector{T}, oper::AbstractOperator{T}, x::AbstractVector{T}, maxlevel::Int) where T = error("Unimplemented")
operator_prolong!(y::AbstractVector{T}, oper::AbstractOperator{T}, x::AbstractVector{T}, maxlevel::Int) where T = error("Unimplemented")