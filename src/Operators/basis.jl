############################
## Exports #################
############################

export Basis, cell_value_stencil, subcell_value_stencil, vertex_value_stencil
export cell_derivative_stencil, subcell_derivative_stencil, vertex_derivative_stencil
export value_stencil
export CellStencil, VertexStencil

export Operator, ValueOperator, operator_stencil

############################
## Stencils ################
############################

"""
Represents a cell centered stencil.
"""
struct CellStencil{T, L, R} 
    left::NTuple{L, T}
    center::T
    right::NTuple{R, T}

    CellStencil(left::NTuple{L, T}, center::T, right::NTuple{R, T}) where {T, L, R} = new{T, L, R}(left, center, right)
end

"""
Represents a vertex centered stencil.
"""
struct VertexStencil{T, L, R} 
    left::NTuple{L, T}
    right::NTuple{R, T}

    CellStencil(left::NTuple{L, T}, right::NTuple{R, T}) where {T, L, R} = new{T, L, R}(left, right)
end

########################
## Operators ###########
########################

"""
An abstract numerical operator.
"""
abstract type Operator end

"""
Represents the rank `R` covariant derivative of a function.
"""
struct ValueOperator{R} <: Operator end

###############################
## Basis ######################
###############################

"""
An abstract function basis for a numerical domain.
"""
abstract type Basis{T} end

"""
Returns the cell-centered stencil for computing the value at a cell.
"""
cell_value_stencil(::Basis, ::Val{L}, ::Val{R}) where {L, R} = error("Unimplemented")

"""
Returns the cell-centered stencil for computing the value at a subcell.
"""
subcell_value_stencil(::Basis, ::Val{S}, ::Val{L}, ::Val{R}) where {S, L, R} = error("Unimplemented")

"""
Returns the vertex-centered stencil for computing the value on a vertex.
"""
vertex_value_stencil(::Basis, ::Val{L}, ::Val{R}) where {L, R} = error("Unimplemented")

"""
Returns the cell-centered stencil for computing the derivative at a cell.
"""
cell_derivative_stencil(::Basis, ::Val{L}, ::Val{R}) where {L, R} = error("Unimplemented")

"""
Returns the cell-centered stencil for computing the derivative at a subcell.
"""
subcell_derivative_stencil(::Basis, ::Val{S}, ::Val{L}, ::Val{R}) where {S, L, R} = error("Unimplemented")

"""
Returns the vertex-centered stencil for computing the derivative on a vertex.
"""
vertex_derivative_stencil(::Basis, ::Val{L}, ::Val{R}) where {L, R} = error("Unimplemented")

"""
Returns the cell-centered, balanced stencil for computing the `R`-th covariant derivative.
"""
value_stencil(::Basis, ::Val{O}, ::Val{R}) = error("Unimplemented")


operator_stencil(::Basis, ::Val{O}, ::Operator) = error("Unimplemented")
operator_stencil(basis::Basis, ::Val{O}, ::ValueOperator{R}) where {O, R} = value_stencil(basis, Val(O), Val(R))

# stencil(basis::Basis, ::CellValueOperator{L, R}) where {L, R} = cell_value_stencil(basis, Val(L), Val(R))
# stencil(basis::Basis, ::SubCellValueOperator{S, L, R}) where {S, L, R} = subcell_value_stencil(basis, Val(S), Val(L), Val(R))
# stencil(basis::Basis, ::VertexValueOperator{L, R}) where {L, R} = vertex_value_stencil(basis, Val(L), Val(R))

# stencil(basis::Basis, ::CellDerivativeOperator{L, R}) where {L, R} = cell_derivative_stencil(basis, Val(L), Val(R))
# stencil(basis::Basis, ::SubCellDerivativeOperator{S, L, R}) where {S, L, R} = subcell_derivative_stencil(basis, Val(S), Val(L), Val(R))
# stencil(basis::Basis, ::VertexDerivativeOperator{L, R}) where {L, R} = vertex_derivative_stencil(basis, Val(L), Val(R))

# stencil(::Basis, ::ValueOperator{O, R}) where {O, R} = value_stencil(::Basis, Val(O), Val(R))