#######################
## Exports ############
#######################

export TransformedFunction, TransformedGradient, TransformedHessian, TransformOperator

#######################
## Transformation #####
#######################

"""
Represents a coordinate transformation applied to an analytic function.
"""
struct TransformedFunction{T<: Transform, F<:AnalyticFunction} <: AnalyticFunction
    transform::T
    func::F
end

(func::TransformedFunction)(x) = func.func(func.transform(x))

"""
Represents a coordinate transformation applied to a analytic gradient.
"""
struct TransformedGradient{T<: Transform, G<:AnalyticGradient} <: AnalyticGradient
    transform::T
    grad::G
end

(func::TransformedGradient)(x) = jacobian(func.transform, x) * func.basis(func.transform(x))

"""
Represents a coordinate transformation applied to a analytic hessian.
"""
struct TransformedHessian{T<: Transform, H<:AnalyticHessian} <: AnalyticHessian
    transform::T
    hess::H
end

(hess::TransformedHessian)(x) = jacobian(hess.transform, x) * jacobian(hess.transform, x) * hess.hess(func.transform(x))

gradient(func::TransformedFunction) = TransformedGradient(func.transform, gradient(func.func))
hessian(func::TransformedFunction) = TransformedHessian(func.transform, hessian(func.func))

"""
An operator which transforms an analytic function and its derivatives
"""
struct TransformOperator{T <: Transform}  <: AnalyticOperator
    transform::T
end

(oper::TransformOperator)(func) = TransformedFunction(oper.transform, func)
(oper::TransformOperator)(func::TransformedFunction) = TransformedFunction(oper.transform ∘ func.transform, func.func)