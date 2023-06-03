##########################
## Exports ###############
##########################

export AnalyticUnknown, AnalyticGradientUnknown, AnalyticHessianUnknown, unknown

##########################
## Unknown ###############
##########################

"""
Represents an unknown function. This is mainly used to construct implicit operators.
"""
struct AnalyticUnknown <: AnalyticFunction end

Base.:(*)(scale, ::AnalyticUnknown) = ValueOperator(scale)

const unknown = AnalyticUnknown()

"""
Represents an unknown gradient. This is mainly used to construct implicit operators.
"""
struct AnalyticGradientUnknown <: AnalyticGradient end

gradient(::AnalyticUnknown) = AnalyticGradientUnknown()

LinearAlgebra.:(⋅)(direction, ::AnalyticGradientUnknown) = GradientOperator(direction)

"""
Represents an unknown hessian. This is mainly used to construct implicit operators.
"""
struct AnalyticHessianUnknown <: AnalyticHessian end

hessian(::AnalyticUnknown) = AnalyticHessianUnknown()

LinearAlgebra.:(⋅)(direction, ::AnalyticHessianUnknown) = HessianOperator(direction)

laplacian(::AnalyticUnknown) = LaplacianOperator()


