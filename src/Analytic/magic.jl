##########################
## Exports ###############
##########################

export AnalyticUnknown, AnalyticGradientUnknown, AnalyticHessianUnknown, unknown

##########################
## Functions #############
##########################

Base.:(*)(scale, func::AnalyticFunction) = Directional(scale, func)
LinearAlgebra.:(⋅)(dir, grad::AnalyticGradient) = Directional(dir, grad)
LinearAlgebra.:(⋅)(dir, hess::AnalyticHessian) = Directional(dir, hess)

##########################
## Operators #############
##########################

"""
Represents an unknown function. This is mainly used to construct implicit operators.
"""
struct AnalyticUnknown <: AnalyticFunction end

Base.:(*)(scale, ::AnalyticUnknown) = ValueOperator(scale)

unknown() = AnalyticUnknown()

"""
Represents an unknown gradient. This is mainly used to construct implicit operators.
"""
struct AnalyticGradientUnknown <: AnalyticGradient end

gradient(::AnalyticUnknown) = AnalyticGradientUnknown()

LinearAlgebra.:(⋅)(direction, ::AnalyticGradientUnknown) = DerivativeOperator(direction)

"""
Represents an unknown hessian. This is mainly used to construct implicit operators.
"""
struct AnalyticHessianUnknown <: AnalyticHessian end

hessian(::AnalyticUnknown) = AnalyticHessianUnknown()

LinearAlgebra.:(⋅)(direction, ::AnalyticHessianUnknown) = CurvatureOperator(direction)

laplacian(::AnalyticUnknown) = LaplacianOperator()


