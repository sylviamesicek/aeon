## Exports
export Domain
export ApproxFunction, ApproxFunctional, ApproxGeneric, ApproxCovariant

## Core 
struct Domain{N, T}
    positions::Vector{SVector{N, T}}

    Domain(positions::Vector{SVector{N, T}}) where {N, T} = new{N, T}(positions)
end

Base.length(domain::Domain) = length(domain.positions)
Base.eachindex(domain::Domain) = eachindex(domain.positions)
Base.getindex(domain::Domain, i) = domain.positions[i]

struct ApproxFunction{N, T} 
    values::Vector{T}
end

abstract type ApproxFunctional{N, T, R} end

(oper::ApproxFunctional{N, T})(func::ApproxFunction{N, T}) where {N, T} = error("Application of $(typeof(oper)) on $(typeof(func)) is undefined.")

struct ApproxGeneric{N, T, R}  <: ApproxFunctional{N, T, R}
    stencil::Vector{R}
end

function (oper::ApproxGeneric{N, T, R})(func::ApproxFunction{N, T}) where {N, T, R}
    result = zero(R)
    for (v, s) in zip(func.values, oper.stencil)
        result += v * s
    end
    result
end

struct ApproxCovariant{N, T, O, L} <: ApproxFunctional{N, T, Covariant{N, T, O, L}}
    # SoA storage
    values::SArray{NTuple{O, N}, Vector{T}, O, L}
end

# Optimize with generated functions
function (oper::ApproxCovariant{N, T, O, L})(func::ApproxFunction{N, T}) where {N, T, O, L}
    tdims = ntuple(_ -> N, Val(O))
    tcoords = CartesianIndices(tdims)
    Covariant(StaticArrays.sacollect(SArray{NTuple{O, N}, Vector{T}, O, L}, dot(func.values, oper.values[coord]) for coord in tcoords))
end