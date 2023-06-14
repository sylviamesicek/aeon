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

abstract type ApproxFunctional{N, T, O} end

(oper::ApproxFunctional{N, T})(func::ApproxFunction{N, T}) where {N, T} = error("Application of $(typeof(oper)) on $(typeof(func)) is undefined.")

struct ApproxScalar{N, T} <: ApproxFunctional{N, T, T}
    stencil::Vector{T}
end

function (oper::ApproxScalar{N, T})(func::ApproxFunction{N, T}) where {N, T}
    result = zero(T)
    for (v, s) in zip(func.values, oper.stencil)
        result += v * s
    end
    result
end

struct ApproxCovariant{N, T, O, L} <: ApproxFunctional{N, T, Covariant{N, T, O, L}}
    stencil::Vector{Covariant{N, T, O, L}}
end

function ApproxCovariant(scount::Int, stencil::STensor{N, Vector{T}, O, L}) where {N, T, O, L}
    mstencil = Vector{Covariant{N, T, O, L}}(undef, scount)

    tdims = ntuple(_ -> N, Val(O))
    tcoords = CartesianIndices(tdims)
    
    for i in 1:scount
        mstencil[i] = Covariant(StaticArrays.sacollect(STensor{N, T, O, L}, stencil[coord][i] for coord in tcoords))
    end

    ApproxCovariant{N, T, O, L}(mstencil)
end

# Optimize with generated functions
function (oper::ApproxCovariant{N, T, O, L})(func::ApproxFunction{N, T}) where {N, T, O, L}
    # tdims = ntuple(_ -> N, Val(O))
    # tcoords = CartesianIndices(tdims)
    # Covariant(StaticArrays.sacollect(STensor{N, Vector{T}, O, L}, dot(func.values, oper.values[coord]) for coord in tcoords))
    @assert length(oper.stencil) == length(func.values)

    result = zero(STensor{N, T, O, L})
    for i in 1:length(oper.stencil)
        result += func.values[i] * oper.stencil[i].inner
    end
    Covariant(result)
end