# Exports

export ACovariant, ADerivative, ACurvature
export ∇, ∇²

# Core

"""
A functional which returns a covariant argument. This mainly includes covariant derivatives of various orders
"""
const ACovariant{N, T, O, L} = AFunctional{N, T, Covariant{N, T, O, L}} where {N, T, O, L}

(oper::ACovariant{N, T, O})(func::AFunction{N, T}, ::SVector{N, T}) where {N, T, O} = error("Covariant functional $(typeof(oper)) on $(typeof(func)) is undefined.")

# const AValue{N, T} = ACovariant{N, T, 0, 1}()
"""
A first-order covariant derivative (also known as a gradient).
"""
struct ADerivative{N, T} <: ACovariant{N, T, 1, N} end

"""
A second-order covariant derivative (also known as a hessian).
"""
struct ACurvature{N, T, L} <: ACovariant{N, T, 2, L} 
    ACurvature{N, T}() where {N, T} = new{N, T, N*N}()
end

"""
Builds a covaraint derivative from the identity operator.
"""
∇(::AIdentity{N, T}) where {N, T} = ADerivative{N, T}()

"""
Builds a covariant 2nd derivative from the identity operator.
"""
∇²(::AIdentity{N, T}) where {N, T} = ACurvature{N, T}()


