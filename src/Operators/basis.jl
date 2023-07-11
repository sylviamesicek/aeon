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

struct CellStencil{T, L, R} 
    left::NTuple{L, T}
    center::T
    right::NTuple{R, T}

    CellStencil(left::NTuple{L, T}, center::T, right::NTuple{R, T}) where {T, L, R} = new{T, L, R}(left, center, right)
end

struct VertexStencil{T, L, R} 
    left::NTuple{L, T}
    right::NTuple{R, T}

    CellStencil(left::NTuple{L, T}, right::NTuple{R, T}) where {T, L, R} = new{T, L, R}(left, right)
end

########################
## Operators ###########
########################

abstract type Operator end

struct ValueOperator{R} <: Operator end

###############################
## Basis ######################
###############################

abstract type Basis{T} end

# Functions to compute arbitrary value stencils with unequal supports
cell_value_stencil(::Basis, ::Val{L}, ::Val{R}) where {L, R} = error("Unimplemented")
subcell_value_stencil(::Basis, ::Val{S}, ::Val{L}, ::Val{R}) where {S, L, R} = error("Unimplemented")
vertex_value_stencil(::Basis, ::Val{L}, ::Val{R}) where {L, R} = error("Unimplemented")

# Functions to compute arbitrary derivative stencils with unequal supports
cell_derivative_stencil(::Basis, ::Val{L}, ::Val{R}) where {L, R} = error("Unimplemented")
subcell_derivative_stencil(::Basis, ::Val{S}, ::Val{L}, ::Val{R}) where {S, L, R} = error("Unimplemented")
vertex_derivative_stencil(::Basis, ::Val{L}, ::Val{R}) where {L, R} = error("Unimplemented")

# Computes the rank R covariant derivative centered cell stencil
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