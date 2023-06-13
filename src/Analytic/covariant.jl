# Exports

export CovariantOperator, DerivativeOperator, CurvatureOperator
export derivative, curvature

struct CovariantOperator{N, T, O, L} <: AnalyticOperator{N, T, SArray{NTuple{O, N}, T, O, L}} 
    CovariantOperator{N, T, O}() where {N, T, O} = new{N, T, O, N^O}()
end

# @generated function (operator::CovariantOperator{N, T})(func::Varargs{AnalyticFunction{N, T}, D}) where {N, T, D}
#     exprs = [:(operator(func[$i])) for i in 1:D]
#     Expr(:call, CombinedFunction, exprs...)
# end

# (operator::CovariantOperator{N, T})(func::ScaledFunction{N, T}) where {N, T} = ScaledFunction(func.scale, operator(func))
# (operator::CovariantOperator{N, T})(func::CombinedFunction{N, T}) where {N, T}= operator(func.inner...)

function (operator::CovariantOperator{N, T, O})(func::TransformedFunction{N, T}) where {N, T, O}
    nfunc = ScaledFunction(jacobian(func.transform)^O, operator(func.inner))
    TransformedFunction(func.transform, nfunc)
end

# const ValueOperator{N, T} = CovariantOperator{N, T, 0, }()
const DerivativeOperator{N, T, L} = CovariantOperator{N, T, 1, L} where {N, T, L}
const CurvatureOperator{N, T, L} = CovariantOperator{N, T, 2, L} where {N, T, L}

DerivativeOperator{N, T}() where {N, T} = CovariantOperator{N, T, 1}
CurvatureOperator{N, T}() where {N, T} = CovariantOperator{N, T, 2}

derivative(::IdentityOperator{N, T}) where {N, T} = DerivativeOperator{N, T}()
curvature(::IdentityOperator{N, T}) where {N, T} = CurvatureOperator{N, T}()

