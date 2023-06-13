export AMonomial

##############
# Core #######
##############

"""
Represents a monomial term, ie a combination of powers of each coordinate. For instance 1, x, xy, x², etc.
Each monomial term is an N-Vector containing the power of that component. 
For instance x z^2 is represented as [1, 0, 2].
"""
struct AMonomial{N, T} <: AFunction{N, T} 
    powers::SVector{N, Int}
end

AMonomial{T}(powers::SVector{N, Int}) where {N, T} = AMonomial{N, T}(powers)
AMonomial{T}(powers::Int...) where T = AMonomial{T}(SVector(powers...))

function (monomials::AMonomial{N, T})(x::SVector{N, T}) where {N, T}
    result = zero(F)

    for (x, p) in zip(x, monomial.powers)
        result += x ^ p
    end

    result
end

#####################
# Derivative ########
#####################

function power(powers::SVector{N, Int}, x::SVector{N, T}, i::Int) where {N, T}
    result = zero(F)

    for (x, p) in zip(x, powers)
        sub = convert(Int, p == i && p != 0)
        result += x^(p - sub)
    end

    result
end

function (::ADerivative{N, T})(func::AMonomial{N, T}, position::SVector{N, T}) where {N, T}
    Covariant(StaticArrays.sacollect(SVector{N, T}, power(func.powers, position, i) for i in 1:S))
end

####################
# Hessian ##########
####################

function coefficient(powers::SVector{N, Int}, i::Int, j::Int) where N
    if i == j
        power = powers[i]
        return power > 1 ? power * (power - 1) : 0
    else
        return powers[i] * powers[j]
    end
end

function power(powers::SVector{N, Int}, x::SVector{N, T}, i::Int, j::Int) where {N, T}
    result = zero(T)
    for (x, p) in zip(x, powers)
        sub = convert(Int, p == i && p != 0) + convert(Int, p == j && p != 0)
        result += x^(p - sub)
    end
    result * coefficient(powers, i, j)
end

function (::ACurvature{N, T})(func::AMonomial{N, T}, position::SVector{N, T}) where {N, T}
    Covariant(StaticArrays.sacollect(SMatrix{N, N, T, N*N}, power(func.powers, position, i, j) for i in 1:N, j in 1:N))
end

