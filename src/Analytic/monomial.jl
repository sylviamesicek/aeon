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

function (monomial::AMonomial{N, T})(x::SVector{N, T}) where {N, T}
    result = one(T)

    for p in 1:N
        result *= x[p]^monomial.powers[p]
    end

    result
end

#####################
# Derivative ########
#####################

function power(powers::SVector{N, Int}, x::SVector{N, T}, i::Int) where {N, T}
    result = powers[i]

    for p in 1:N
        sub = ifelse(p == i && powers[p] != 0, 1, 0)
        result *= x[p]^(powers[p] - sub)
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
    powers[i] * (powers[j] - ifelse(i == j, 1, 0))
end

function power(powers::SVector{N, Int}, x::SVector{N, T}, i::Int, j::Int) where {N, T}
    result = coefficient(powers, i, j)
    for p in 1:N
        sub = ifelse(p == i && powers[p] > 1, 1, 0) + ifelse(p == j && powers[p] > 1, 1, 0)
        result *= x[p]^(powers[p] - sub)
    end
    result
end

function (::ACurvature{N, T})(func::AMonomial{N, T}, position::SVector{N, T}) where {N, T}
    Covariant(StaticArrays.sacollect(SMatrix{N, N, T, N*N}, power(func.powers, position, i, j) for i in 1:N, j in 1:N))
end

