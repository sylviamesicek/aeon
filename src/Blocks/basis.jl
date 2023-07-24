############################
## Exports #################
############################

export Stencil, stencil_diagonal
export AbstractBasis, cell_value_stencil, subcell_value_stencil, vertex_value_stencil
export cell_derivative_stencil, subcell_derivative_stencil, vertex_derivative_stencil
export value_stencil

############################
## Stencils ################
############################

"""
Represents a stencil to be applied to a uniform grid
"""
struct Stencil{T, L, R}
    left::NTuple{L, T}
    center::T
    right::NTuple{R, T}

    Stencil(left::NTuple{L, T}, center::T, right::NTuple{R, T}) where {T, L, R} = new{T, L, R}(left, center, right)
end

Base.show(io::IO, stencil::Stencil) = print(io, "Stencil($(reverse(stencil.left)), $(stencil.center), $(stencil.right))")
Base.:(-)(stencil::Stencil) = Stencil(.-stencil.left, -stencil.center, .-stencil.right)

###############################
## Diagonal ###################
###############################

"""
Computes the diagnol element of a stencil product.
"""
function stencil_diagonal(stencils::NTuple{N, Stencil{T}}) where {N, T}
    prod(map(s -> s.center, stencils))
end

###############################
## Basis ######################
###############################

"""
An abstract function basis for a numerical domain.
"""
abstract type AbstractBasis{T} end

"""
Returns the cell-centered stencil for computing the value at a cell.
"""
cell_value_stencil(::AbstractBasis, ::Val{L}, ::Val{R}) where {L, R} = error("Unimplemented")

"""
Returns the cell-centered stencil for computing the value at a subcell.
"""
subcell_value_stencil(::AbstractBasis, ::Val{L}, ::Val{R}, ::Val{S}) where {S, L, R} = error("Unimplemented")

"""
Returns the vertex-centered stencil for computing the value on a vertex.
"""
vertex_value_stencil(::AbstractBasis, ::Val{L}, ::Val{R}, ::Val{S}) where {L, R, S} = error("Unimplemented")

"""
Returns the cell-centered stencil for computing the derivative at a cell.
"""
cell_derivative_stencil(::AbstractBasis, ::Val{L}, ::Val{R}) where {L, R} = error("Unimplemented")

"""
Returns the cell-centered stencil for computing the derivative at a subcell.
"""
subcell_derivative_stencil(::AbstractBasis, ::Val{L}, ::Val{R}, ::Val{S}) where {S, L, R} = error("Unimplemented")

"""
Returns the vertex-centered stencil for computing the derivative on a vertex.
"""
vertex_derivative_stencil(::AbstractBasis, ::Val{L}, ::Val{R}, ::Val{S}) where {S, L, R} = error("Unimplemented")

"""
Returns the cell-centered, balanced stencil for computing the `R`-th covariant derivative.
"""
value_stencil(::AbstractBasis, ::Val{O}, ::Val{R}) where {O, R} = error("Unimplemented")