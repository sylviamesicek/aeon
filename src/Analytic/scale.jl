########################
## Exports #############
########################

export ScaledFunction, ScaledGradient, ScaledHessian, ScaleOperator

########################
## Scaling #############
########################

struct ScaledFunction{F, Func <: AnalyticFunction} <:AnalyticFunction
    scale::F
    func::Func
end

(func::ScaledFunction)(x::AbstractVector) = func.scale * func.func(x)

struct ScaledGradient{F, G <: AnalyticGradient} <: AnalyticGradient
    scale::F
    grad::G
end

(func::ScaledGradient)(x::AbstractVector) = func.scale * func.grad(x)

struct ScaledHessian{F, H <: AnalyticHessian} <: AnalyticHessian
    scale::F
    hess::H
end

(func::ScaledHessian)(x::AbstractVector) = func.scale * func.hess(x)

gradient(func::ScaledFunction) = ScaledGradient(func.scale, gradient(func.func))
hessian(func::ScaledFunction) = ScaledHessian(func.scale, hessian(func.func))

struct ScaleOperator{F} <: AnalyticOperator
    scale::F
end

(oper::ScaleOperator)(func::AnalyticFunction) = ScaledFunction(oper.scale, func)
(oper::ScaleOperator)(func::ScaledFunction) = ScaledFunction(oper.scale ∘ func.scale, func.func)
