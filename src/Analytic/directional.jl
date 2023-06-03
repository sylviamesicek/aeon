####################
## Exports #########
####################

export ValueOperator, GradientOperator, HessianOperator, LaplacianOperator

####################
## Direction #######
####################

"""
Represents a directional function. That is, a function which takes the dot product of its output with a direction scalar, vector, or matrix.
"""
struct Directional{D, F} <: AnalyticFunction
    direction::D
    func::F
end

(func::Directional)(x::AbstractVector) = dot(func.direction, func.func(x))

####################
## Operators #######
####################

"""
Represents a simple scaling of an analytic function.
"""
struct ValueOperator{F} <: AnalyticOperator
    scale::F
end

(oper::ValueOperator)(func::AnalyticFunction) = Directional(oper.scale, func)

"""
Represents the directional derivative of an analytic function.
"""
struct GradientOperator{F} <: AnalyticOperator
    direction::F
end

(oper::GradientOperator)(func::AnalyticFunction) = Directional(oper.direction, gradient(func))

"""
Represents the directional hessian of an analytic function.
"""
struct HessianOperator{F} <: AnalyticOperator
    direction::F
end

(oper::HessianOperator)(func::AnalyticFunction) = Directional(oper.direction, hessian(func))

"""
Represents applying a flat laplacian to an analytic function
"""
struct LaplacianOperator <: AnalyticOperator
end

(oper::LaplacianOperator)(func::AnalyticFunction) = Directional(I, hessian(func))
