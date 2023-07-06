#############################
## Exports ##################
#############################

export CellStencil, VertexStencil, ProlongStencil, BoundaryCoefs, InterfaceCoefs, order
export Operator, cell_stencil, vertex_stencil, cell_left_stencil, cell_right_stencil, boundary_value_stencil, boundary_derivative_stencil
export LagrangeOperator, LagrangeValue, LagrangeDerivative, LagrangeDerivative2

##############################
## Stencils ##################
##############################

struct CellStencil{T, O} 
    left::NTuple{O, T}
    center::T
    right::NTuple{O, T}

    CellStencil(left::NTuple{O, T}, center::T, right::NTuple{O, T}) where {O, T} = new{O, T}(left, center, right)
end

order(::CellStencil{T, O}) where {T, O} = O

struct VertexStencil{T, O} 
    left::NTuple{O, T}
    right::NTuple{O, T}

    VertexStencil(left::NTuple{O, T}, right::NTuple{O, T}) where {O, T} = new{T, O}(left, right)
end

order(::VertexStencil{T, O}) where {T, O} = O

struct BoundaryCoefs{T, O, L}
    interior::NTuple{O, T}
    exterior::NTuple{L, T}
end

struct InterfaceCoefs{T, L, O} 
    values::NTuple{L, T}

    InterfaceCoefs{O}(values::NTuple{L, T}) where {L, T, O} = new{T, L, O}(values) 
end

function InterfaceCoefs(stencil::CellStencil{T, O}, side::Bool, length::Int) where {T, O}
    if side
        return InterfaceCoefs{O}(ntuple(i -> stencil.right[end-length+i], length))
    else
        return InterfaceCoefs{O}(ntuple(i -> stencil.left[end-length+i], length))
    end
end

function InterfaceCoefs(stencil::VertexStencil{T, O}, side::Bool, length::Int) where {T, O}
    if side
        return InterfaceStencil{O}(ntuple(i -> stencil.right[end-length+i], length))
    else
        return InterfaceStencil{O}(ntuple(i -> stencil.left[end-length+i], length))
    end
end

#############################
## Operators ################
#############################

abstract type Operator{T} end

# odd_stencil_order(::Operator) = error("Unimplemented")
# even_stencil_order(::Operator) = error("Unimplemented")

cell_stencil(::Operator) = error("Unimplemented")
cell_left_stencil(::Operator) = error("Unimplemented")
cell_right_stencil(::Operator) = error("Unimplemented")
vertex_stencil(::Operator) = error("Unimplemented")

# A boundary value stencil compatible with the operator
boundary_value_coefs(::Operator, i::Int, side::Bool) = error("Unimplemented")
boundary_derivative_coefs(::Operator, i::Int, side::Bool) = error("Unimplemented")

############################
## Generic #################
############################

cell_centered_grid(L, R) = ntuple(L + R + 1) do i
    i - L + 0//1
end

vertex_centered_grid(L, R) = ntuple(L + R + 1) do i
    i - L + 1//2
end

############################
## Lagrange stencils #######
############################

abstract type LagrangeOperator{T, O} <: Operator{T} end

function boundary_value_coefs(::LagrangeOperator{T, O}, i::Int, side::Bool) where {T, O}
    L = ifelse(side, O, i)
    R = ifelse(side, i, O)

    grid = vertex_centered_grid(L, R)
    stencil = lagrange(grid, 0//1)

    left = reverse(ntuple(i -> T(stencil[i]), L))
    right = ntuple(i -> T(stencil[L + i]), R)

    BoundaryCoefs(ifelse(side, left, right), ifelse(side, right, left))
end

function boundary_derivative_coefs(::LagrangeOperator{T, O}, i::Int, side::Bool) where {T, O}
    L = ifelse(side, O, i)
    R = ifelse(side, i, O)

    grid = vertex_centered_grid(L, R)
    stencil = lagrange_derivative(grid, 0//1)

    left = reverse(ntuple(i -> T(stencil[i]), L))
    right = ntuple(i -> T(stencil[L + i]), R)

    BoundaryCoefs(ifelse(side, left, right), ifelse(side, right, left))
end


# Helpers
function lagrange_cell_stencil(stencil::NTuple{L, T}) where {L, T}
    O = (L - 1) ÷ 2

    left = reverse(ntuple(i -> T(stencil[i]), O))
    right = ntuple(i -> T(stencil[i + 1 + O]), O)

    CellStencil(left, stencil[O + 1], right)
end

function lagrange_vertex_stencil(stencil::NTuple{L, T}) where {L, T}
    O = L ÷ 2

    left = reverse(ntuple(i -> T(stencil[i]), O))
    right = ntuple(i -> T(stencil[i + O]), O)

    VertexStencil(left, right)
end

######################
## Lagrange Value ####
######################

struct LagrangeValue{T, O} <: LagrangeOperator{T, O} end

function cell_stencil(::LagrangeValue{T})
    CellStencil((), one(T), ())
end

function cell_left_stencil(::LagrangeValue{T}) where {T}
    grid = cell_centered_grid(O, O)
    stencil = map(T, lagrange(grid, -1//2))
    lagrange_cell_stencil(stencil)
end

function cell_right_stencil(::LagrangeValue{T}) where {T}
    grid = cell_centered_grid(O, O)
    stencil = map(T, lagrange(grid, 1//2))
    lagrange_cell_stencil(stencil)
end

function vertex_stencil(::LagrangeValue{T, O}) where {T, O}
    grid = vertex_centered_grid(O + 1, O + 1)
    stencil = map(T, lagrange(grid, 0//1))
    lagrange_vertex_stencil(stencil)
end

############################
## Lagrange Derivative #####
############################

struct LagrangeDerivative{T, O} <: LagrangeOperator{T, O} end

function cell_stencil(::LagrangeDerivative{T}) where {T}
    grid = cell_centered_grid(O, O)
    stencil = map(T, lagrange_derivative(grid, 0//1))
    lagrange_cell_stencil(stencil)
end

function cell_left_stencil(::LagrangeDerivative{T}) where {T}
    grid = cell_centered_grid(O, O)
    stencil = map(T, lagrange_derivative(grid, -1//2))
    lagrange_cell_stencil(stencil)
end

function cell_right_stencil(::LagrangeDerivative{T}) where {T}
    grid = cell_centered_grid(O, O)
    stencil = map(T, lagrange_derivative(grid, 1//2))
    lagrange_cell_stencil(stencil)
end

function vertex_stencil(::LagrangeDerivative{T, O}) where {T, O}
    grid = vertex_centered_grid(O + 1, O + 1)
    stencil = map(T, lagrange_derivative(grid, 0//1))
    lagrange_vertex_stencil(stencil)
end

############################
## Lagrange Derivative 2 ###
############################

struct LagrangeDerivative2{T, O} <: LagrangeOperator{T, O} end

function cell_stencil(::LagrangeDerivative2{T}) where {T}
    grid = cell_centered_grid(O, O)
    stencil = lagrange_derivative_2(grid, 0//1)
    lagrange_cell_stencil(stencil)
end

function cell_left_stencil(::LagrangeDerivative2{T}) where {T}
    grid = cell_centered_grid(O, O)
    stencil = map(T, lagrange_derivative_2(grid, -1//2))
    lagrange_cell_stencil(stencil)
end

function cell_right_stencil(::LagrangeDerivative2{T}) where {T}
    grid = cell_centered_grid(O, O)
    stencil = map(T, lagrange_derivative_2(grid, 1//2))
    lagrange_cell_stencil(stencil)
end

function vertex_stencil(::LagrangeDerivative2{T, O}) where {T, O}
    grid = vertex_centered_grid(O + 1, O + 1)
    stencil = lagrange_derivative_2(grid, 0//1)
    lagrange_vertex_stencil(stencil)
end

function prolong_stencil(::LagrangeDerivative2{T, O}) where {T, O}
    grid = cell_centered_grid(O, O)
    left = lagrange_derivative_2(grid, -1//2)
    right = lagrange_derivative_2(grid, 1//2)
    ProlongStencil(lagrange_cell_stencil(left), lagrange_cell_stencil(right))
end