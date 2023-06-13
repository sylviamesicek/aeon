# Exports

export ValueOperator, DerivativeOperator, CurvatureOperator
export derivative, curvature

abstract type AnalyticCovariant{N, T, O, R} <: AnalyticOperator{N, T, R} end

# Operator

struct ValueOperator{N, T} <: AnalyticCovariant{N, T, 0, T} end

(operator::ValueOperator{N, T})(field::AnalyticFunction{N, T}) where {N, T} = field

struct DerivativeOperator{N, T} <: AnalyticCovariant{N, T, 1, SVector{N, T}} end

derivative(::ValueOperator{N, T}) where {N, T} = DerivativeOperator{N, T}()

struct CurvatureOperator{N, T, L} <: AnalyticCovariant{N, T, 2, SMatrix{N, N, T, L}} 
    CurvatureOperator{N, T}() where {N, T} = new{N, T, N*N}()
end

curvature(::ValueOperator{N, T}) where {N, T} = CurvatureOperator{N, T}()

