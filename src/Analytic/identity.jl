########################
## Exports #############
########################

export IdentityOperator, IdentityFunction

########################
## Identity ############
########################

"""
Represents the identity or `value` operator. Given a function, it returns the same function.
"""
struct IdentityOperator <: AnalyticOperator end

(::IdentityOperator)(func::AnalyticOperator) = func

struct IdentityFunction <: AnalyticFunction end

(::IdentityFunction)(::AbstractVector{F}) where {F} = one(F)