export Stencil, LeftStencil, RightStencil, CenteredStencil
export ProlongedOddStencil, ProlongedEvenStencil, RestrictedStencil
export stencil_product

####################
## Stencil #########
####################

"""
Represents a stencil. Aka, a slice of coefficient values along an axis, with a method to convert from
"stencil indices" to global indices. 
"""
abstract type Stencil{T} end

"""
Returns the number of non-zero coefficients in stencil
"""
function stencil_length end

"""
Returns an iterator over the indices of a stencil
"""
function stencil_indices end

"""
Returns the value of a stencil at a specific index
"""
function stencil_value end

"""
Converts a stencil index to a global index, given a global row index, and the total number of columns along that axis.
"""
function stencil_to_global end

"""
The axis on which a stencil is defined.
"""
function stencil_axis end

###############################
## Helpers ####################
###############################

"""
Converts the length of a centered stencil to its correct offset, regardless of whether it is odd or even.
"""
centered_offset(length) = (length + 1) ÷ 2

"""
Converts a coarse index to its corresponding refined index
"""
coarse_to_refined(n) = 2n - 1

"""
Converts a refined index to its corresponding coarse index. If n is even, this function treats it as if it is an odd index one lower.
"""
refined_to_coarse(n) = (n + 1) ÷ 2

###############################
## Left #######################
###############################

"""
A stencil on the left side of a numerical domain.
"""
struct LeftStencil{T, L} <: Stencil{T}
    values::SVector{T, L}
    axis::Int

    function LeftStencil(values::SVector{L, T}, axis::Int) where {T, L}
        new{T, L}(values, axis)
    end
end

stencil_length(left::LeftStencil) = length(left.values)
stencil_indices(left::LeftStencil) = eachindex(left.values)
stencil_value(left::LeftStencil, index::Int) = left.values[index]
stencil_to_global(left::LeftStencil, row::Int, total::Int, index::Int) = index
stencil_axis(left::LeftStencil) = left.axis

"""
A stencil on the right side of a numerical domain.
"""
struct RightStencil{T, L} <: Stencil{T}
    values::SVector{L, T}
    axis::Int

    function RightStencil(values::SVector{L, T}, axis::Int) where {T, L}
        new{T, L}(values, axis)
    end
end

stencil_length(right::RightStencil) = length(right.values)
stencil_indices(right::RightStencil) = eachindex(right.values)
stencil_value(right::RightStencil, index::Int) = right.values[index]
stencil_to_global(right::RightStencil, row::Int, total::Int, index::Int) = total - length(right.values) + index
stencil_axis(stencil::RightStencil) = stencil.axis

"""
A stencil with equal numbers of support points on each side, that may be used on the interior of a domain.
"""
struct CenteredStencil{T, L} <: Stencil{T}
    values::SVector{T, L}
    axis::Int

    function CenteredStencil(values::SVector{T, L}, axis::Int) where {T, L}
        @assert L % 2 == 1
        new{T, O, L}(values, axis)
    end
end

stencil_length(center::CenteredStencil) = length(center.values)
stencil_indices(center::CenteredStencil) = eachindex(center.values)
stencil_value(center::CenteredStencil, index::Int) = center.values[index]
stencil_to_global(::CenteredStencil{T, L}, row::Int, total::Int, index::Int) where {T, L} = row - centered_offset(L) + index
stencil_axis(stencil::CenteredStencil) = stencil.axis

"""
A prolonged stencil on an odd point (a point which exists on both the coarse and refined grid). Essentially
an identity stencil that does some manipulation to find the appropiate column to apply the stencil to.
"""
struct ProlongedOddStencil{T} <: Stencil{T} 
    axis::Int

    ProlongedOddStencil{T}(axis::Int) where T = new{T}(axis)
end

stencil_length(::ProlongedOddStencil) = 1
stencil_indices(::ProlongedOddStencil) = 1:1
stencil_value(::ProlongedOddStencil{T}, index::Int) where T = T(1)
stencil_to_global(:ProlongedOddStencil{T}, row::Int, total::Int, index::Int) where {T} = refined_to_coarse(row)
stencil_axis(stencil::ProlongedOddStencil) = stencil.axis

"""
A prolonged stencil on an even point (a point which only exists on the refined mesh).
"""
struct ProlongedEvenStencil{T, L} <: Stencil{T}
    values::SVector{L, T}
    axis::Int

    function ProlongedEvenStencil(values::SVector{L, T}, axis::Int) where {T, L}
        @assert L % 2 == 0
        new{T, L}(values, axis)
    end
end

stencil_length(prol::ProlongedEvenStencil) = length(prol.values) 
stencil_indices(prol::ProlongedEvenStencil) = eachindex(prol.values)
stencil_value(prol::ProlongedEvenStencil, index::Int) = prol.values[index]
stencil_to_global(::ProlongedEvenStencil{T, L}, row::Int, total::Int, index::Int) where {T, L} = refined_to_coarse(row) - centered_offset(L) + index
stencil_axis(stencil::ProlongedEvenStencil) = stencil.axis

struct RestrictedStencil{T, L} <: Stencil{T}
    values::SVector{L, T}
    axis::Int

    function RestrictedStencil(values::SVector{L, T}, axis::Int) where {T, L}
        @assert L % 2 == 1
        new{T, L}(values, axis)
    end
end

stencil_length(stencil::RestrictedStencil) = length(stencil.values) 
stencil_indices(stencil::RestrictedStencil) = eachindex(stencil.values)
stencil_value(stencil::RestrictedStencil, index::Int) = stencil.values[index]
stencil_to_global(::RestrictedStencil{T, L}, row::Int, total::Int, index::Int) where {T, L} = coarse_to_refined(row) - centered_offset(L) + index
stencil_axis(stencil::RestrictedStencil) = stencil.axis

####################
## Product #########
####################

function stencil_product(point::CartesianIndex{N}, func::AbstractArray{T, N}, stencils::NTuple{L, Stencil{T}}) where {N, T, L}
    # Cache tuple from full to stencil dimensions
    full_to_stencil = ntuple(i -> 0, Val(L))

    for sdim in 1:L
        full_to_stencil = setindex(full_to_stencil, sdim, stencil_axis(stencils[sdim]))
    end

    # Get local dims
    localdims = ntuple(Val(L)) do sdim
        axis = stencil_axis(stencils[sdim])
        stencil_length(stencils[sdim])
    end

    # Accumulate result
    result = zero(T)

    for localindex in CartesianIndices(localdims)
        coefficient = one(T)

        for sdim in 1:L
            axis = stencil_axis(stencils[sdim])
            coefficient *= stencil_value(stencils[sdim], localindex[sdim])
        end

        # Get global index into function
        globals = ntuple(Val(N)) do dim
            sdim = full_to_stencil[dim]
            if sdim > 0
                stencil_to_global(stencils[sdim], point[dim], size(func)[dim], localindex[sdim])
            else
                point[dim]
            end
        end

        result += coefficient * func[globals...]
    end

    result
end
