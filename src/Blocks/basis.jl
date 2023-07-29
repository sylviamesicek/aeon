############################
## Stencils ################
############################

export AbstractStencil, CellValue, VertexValue, SubCellValue, CellDerivative, VertexDerivative, SubCellDerivative, CovariantDerivative
export Stencil, stencil_diagonal

"""
A stencil defined independently of basis.
"""
abstract type AbstractStencil{L, R} end

struct CellValue{L, R} <: AbstractStencil{L, R} end
struct VertexValue{L, R, S} <: AbstractStencil{L, R} end
struct SubCellValue{L, R, S} <: AbstractStencil{L, R} end

struct CellDerivative{L, R} <: AbstractStencil{L, R} end
struct VertexDerivative{L, R, S} <: AbstractStencil{L, R} end
struct SubCellDerivative{L, R, S} <: AbstractStencil{L, R} end

struct CovariantDerivative{O, R} <: AbstractStencil{O, O} end


"""
Represents a stencil to be applied to a uniform grid
"""
struct Stencil{T, L, R}
    left::NTuple{L, T}
    center::T
    right::NTuple{R, T}

    """
    Builds a concrete stencil from a set of coefficients.
    """
    Stencil(left::NTuple{L, T}, center::T, right::NTuple{R, T}) where {T, L, R} = new{T, L, R}(left, center, right)
end

Base.show(io::IO, stencil::Stencil) = print(io, "Stencil($(reverse(stencil.left)), $(stencil.center), $(stencil.right))")
Base.:(-)(stencil::Stencil) = Stencil(.-stencil.left, -stencil.center, .-stencil.right)

###############################
## Diagonal ###################
###############################

"""
Computes the diagonal element of a stencil product.
"""
stencil_diagonal(stencils::NTuple{N, Stencil{T}}) where {N, T} = prod(map(s -> s.center, stencils))

###############################
## Basis ######################
###############################

export AbstractBasis

"""
An abstract function basis for a numerical domain.
"""
abstract type AbstractBasis{T} end

"""
Returns the stencil which applies the operator in the given basis.
"""
Stencil(basis::AbstractBasis, operator::AbstractStencil) = error("Stencil is unimplemented for $(typeof(basis)) and $(typeof(operator))")