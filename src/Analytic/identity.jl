########################
## Exports #############
########################

export IdentityOperator

########################
## Identity ############
########################

"""
Represents the identity or `value` operator. Given a function, it returns the same function.
"""
struct IdentityOperator <: AnalyticOperator end

(::IdentityOperator)(func::AnalyticOperator) = func