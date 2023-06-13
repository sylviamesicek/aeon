# Exports

export ACovariant, ADerivative, ACurvature
export ∇, ∇²

# Core

const ACovariant{N, T, O, L} = AFunctional{N, T, Covariant{N, T, O, L}} where {N, T, O, L}

(oper::ACovariant{N, T, O})(func::AFunction{N, T}, ::SVector{N, T}) where {N, T, O} = error("Covariant functional $(typeof(oper)) on $(typeof(func)) is undefined.")

# const AValue{N, T} = ACovariant{N, T, 0, 1}()
struct ADerivative{N, T} <: ACovariant{N, T, 1, N} end

struct ACurvature{N, T, L} <: ACovariant{N, T, 2, L} 
    ACurvature{N, T}() where {N, T} = new{N, T, N*N}
end


∇(::AIdentity{N, T}) where {N, T} = ADerivative{N, T}()
∇²(::AIdentity{N, T}) where {N, T} = ACurvature{N, T}()


