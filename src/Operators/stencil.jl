export Stencil, LeftStencil, RightStencil, CenteredStencil, ProlongedStencil
export stencil_product


abstract type Stencil{T} end

struct LeftStencil{T, BL} <: Stencil{T}
    values::SVector{T, BL}
end

stencil_length(point::Int, left::LeftStencil) = length(left.values)
stencil_indices(point::Int, left::LeftStencil) = eachindex(left.values)
stencil_value(point::Int, left::LeftStencil, index::Int) = left.values[index]
stencil_to_global(point::Int, left::LeftStencil, total::Int, index::Int) = index

struct RightStencil{T, BL} <: Stencil{T}
    values::SVector{T, BL}
end

stencil_length(point::Int,right::RightStencil) = length(right.values)
stencil_indices(point::Int,right::RightStencil) = eachindex(right.values)
stencil_value(point::Int, right::RightStencil, index::Int) = right.values[index]
stencil_to_global(point::Int, right::RightStencil, total::Int, index::Int) = total - length(right.values) + index

struct CenteredStencil{T, O, L} <: Stencil{T}
    values::SVector{T, L}

    function CenteredStencil{T, O}(values::SVector{T, L}) where {T, O, L}
        @assert 2O + 1 == L
        new{T, O, L}(values)
    end

    function CenteredStencil{T}(values::SVector{T, L}) where {T, L}
        @assert L % 2 == 1
        new{T, (L - 1)/2, L}(values)
    end
end

stencil_length(point::Int, center::CenteredStencil) = length(center.values)
stencil_indices(point::Int, center::CenteredStencil) = eachindex(center.values)
stencil_value(point::Int, center::CenteredStencil, index::Int) = center.values[index]
stencil_to_global(point::Int, center::CenteredStencil{T, O}, total::Int, index::Int) where {T, O} = point - O + index

struct IdentityStencil{T} <: Stencil{T} end

stencil_length(point::Int, ::IdentityStencil) = 1
stencil_indices(point::Int, ::IdentityStencil) = 1:1
stencil_value(point::Int, ::IdentityStencil, index::Int) where T = T(1)
stencil_to_global(point::Int, ::IdentityStencil{T}, total::Int, index::Int) where {T} = point

struct ProlongedStencil{T, O, L} <: Stencil{T}
    values::SVector{T, L}

    function ProlongedStencil{T, O}(values::SVector{T, L}) where {T, O, L}
        @assert 2O + 1 == L
        new{T, O, L}(values)
    end

    function ProlongedStencil{T}(values::SVector{T, L}) where {T, L}
        @assert L % 2 == 1
        new{T, (L - 1)/2, L}(values)
    end
end

stencil_length(point::Int, prol::ProlongedStencil) = point % 2 ? length(prol.values) : 1
stencil_indices(point::Int, prol::ProlongedStencil) = point % 2  ? eachindex(prol.values) : 1:1
stencil_value(point::Int, prol::ProlongedStencil{T}, index::Int) where T= point % 2  ? prol.values[index] : T(1)
stencil_to_global(point::Int, prol::ProlongedStencil{T, O}, total::Int, index::Int) where {T, O} = (point + 1) ÷ 2 - O + index

function stencil_product(point::CartesianIndex{N}, func::AbstractArray{T, N}, stencils::NTuple{L, Tuple{Stencil{T}, Int}}) where {N, T, L}
    # Cache tuple from full to stencil dimensions
    full_to_stencil = ntuple(i -> 0, Val(L))

    for dim in 1:L
        full_to_stencil = setindex(full_to_stencil, stencils[dim][2], dim)
    end

    # Get local dims
    localdims = ntuple(Val(L)) do dim
        axis = stencils[dim][2]
        stencil_length(point[axis], stencils[dim][1])
    end

    # Accumulate result
    result = zero(T)

    for localindex in CartesianIndices(localdims)
        coefficient = one(T)

        for dim in 1:L
            axis = stencils[dim][2]
            coefficient *= stencil_value(point[axis], stencils[dim][1], localindex[dim])
        end

        # Get global index into function
        globals = ntuple(Val(N)) do dim
            sdim = full_to_stencil[dim]
            if full_to_stencil[dim] > 0
                stencil_to_global(point[dim], stencils[sdim][1], size(func)[dim], localindex[sdim])
            else
                point[dim]
            end
        end

        result += coefficient * func[globals...]
    end

    result
end
