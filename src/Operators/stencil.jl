export refined_to_coarse, coarse_to_refined

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

index_from_right(total, index) = total - index + 1

###############################
## Left #######################
###############################

"""
A stencil on the left side of a numerical domain.
"""
struct LeftStencil{T, L} <: Stencil{T}
    values::SVector{L, T}

    function LeftStencil(values::SVector{L, T}) where {T, L}
        new{T, L}(values)
    end
end

stencil_length(left::LeftStencil) = length(left.values)
stencil_value(left::LeftStencil, index::Int) = left.values[index]
stencil_to_global(left::LeftStencil, row::Int, index::Int) = index

"""
A stencil on the right side of a numerical domain.
"""
struct RightStencil{T, L} <: Stencil{T}
    values::SVector{L, T}
    total::Int

    function RightStencil(values::SVector{L, T}, total::Int) where {T, L}
        new{T, L}(values, total)
    end
end

stencil_length(right::RightStencil) = length(right.values)
stencil_value(right::RightStencil, index::Int) = right.values[index]
stencil_to_global(stencil::RightStencil, row::Int, index::Int) = index_from_right(stencil.total, index)

"""
A stencil with equal numbers of support points on each side, that may be used on the interior of a domain.
"""
struct CenteredStencil{T, L} <: Stencil{T}
    values::SVector{L, T}

    function CenteredStencil(values::SVector{L, T}) where {T, L}
        @assert L % 2 == 1
        new{T, L}(values)
    end
end

stencil_length(center::CenteredStencil) = length(center.values)
stencil_value(center::CenteredStencil, index::Int) = center.values[index]
stencil_to_global(::CenteredStencil{T, L}, row::Int, index::Int) where {T, L} = row - centered_offset(L) + index

"""
A prolonged stencil on an odd point (a point which exists on both the coarse and refined grid). Essentially
an identity stencil that does some manipulation to find the appropiate column to apply the stencil to.
"""
struct ProlongedOddStencil{T} <: Stencil{T} end

stencil_length(::ProlongedOddStencil) = 1
stencil_value(::ProlongedOddStencil{T}, index::Int) where T = T(1)
stencil_to_global(::ProlongedOddStencil{T}, row::Int, index::Int) where {T} = refined_to_coarse(row)

"""
A prolonged stencil on an even point (a point which only exists on the refined mesh).
"""
struct ProlongedEvenStencil{T, L} <: Stencil{T}
    values::SVector{L, T}

    function ProlongedEvenStencil(values::SVector{L, T}) where {T, L}
        @assert L % 2 == 0
        new{T, L}(values)
    end
end

stencil_length(prol::ProlongedEvenStencil) = length(prol.values) 
stencil_value(prol::ProlongedEvenStencil, index::Int) = prol.values[index]
stencil_to_global(::ProlongedEvenStencil{T, L}, row::Int, index::Int) where {T, L} = refined_to_coarse(row) - centered_offset(L) + index

struct RestrictedStencil{T, L} <: Stencil{T}
    values::SVector{L, T}

    function RestrictedStencil(values::SVector{L, T}) where {T, L}
        @assert L % 2 == 1
        new{T, L}(values)
    end
end

stencil_length(stencil::RestrictedStencil) = length(stencil.values) 
stencil_value(stencil::RestrictedStencil, index::Int) = stencil.values[index]
stencil_to_global(::RestrictedStencil{T, L}, row::Int, index::Int) where {T, L} = coarse_to_refined(row) - centered_offset(L) + index

####################
## Value ###########
####################

struct ValueStencil{T} <: Stencil{T}
    value::T

    ValueStencil(value::T) where T = new{T}(value)
end

stencil_length(stencil::ValueStencil) = 1
stencil_value(stencil::ValueStencil, index::Int) = stencil.value
stencil_to_global(::ValueStencil{T}, row::Int, index::Int) where {T} = row

struct IdentityStencil{T} <: Stencil{T} end

stencil_length(stencil::IdentityStencil) = 1
stencil_value(stencil::IdentityStencil{T}, index::Int) where T = one(T)
stencil_to_global(::IdentityStencil{T}, row::Int, index::Int) where {T} = row

####################
## Product #########
####################

function product(point::CartesianIndex{N}, stencils::NTuple{N, Stencil{T}}, func::AbstractArray{T, N}) where {N, T}
    localdims = ntuple(Val(N)) do dim
        @inline stencil_length(stencils[dim])
    end

    result = zero(T)

    ## Testing
    # kernal = Array{T, N}(undef, localdims...)
    # positions = Array{NTuple{N, T}, N}(undef, localdims...)

    for localindex in CartesianIndices(localdims)
        coefs = ntuple(Val(N)) do dim
            @inline stencil_value(stencils[dim], localindex[dim])
        end

        globals = ntuple(Val(N)) do dim
            @inline stencil_to_global(stencils[dim], point[dim], localindex[dim])
        end

        result += prod(coefs) * func[globals...]

        # kernal[localindex] = prod(coefs)
        # positions[localindex] = globals
    end

    # display(kernal)
    # display(positions)

    result
end