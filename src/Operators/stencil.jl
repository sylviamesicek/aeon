#############################
## Exports ##################
#############################

export CellStencil, VertexStencil, InterfaceStencil
export Operator, cell_stencil, vertex_stencil, cell_left_stencil, cell_right_stencil, interface_value_stencil, interface_derivative_stencil
export LagrangeOperator, LagrangeValue, LagrangeDerivative, LagrangeDerivative2

##############################
## Stencils ##################
##############################

"""
A cell centered stencil that extends `O` in each direction.
"""
struct CellStencil{T, O} 
    left::NTuple{O, T}
    center::T
    right::NTuple{O, T}

    CellStencil(left::NTuple{O, T}, center::T, right::NTuple{O, T}) where {O, T} = new{T, O}(left, center, right)
end

Base.show(io::IO, stencil::CellStencil) = print(io, "Cell Centered Stencil $((reverse(stencil.left)..., stencil.center, stencil.right...))")

"""
A vertex centered stencil that extends `O` cells in each direction.
"""
struct VertexStencil{T, O} 
    left::NTuple{O, T}
    right::NTuple{O, T}

    VertexStencil(left::NTuple{O, T}, right::NTuple{O, T}) where {O, T} = new{T, O}(left, right)
end

Base.show(io::IO, stencil::VertexStencil) = print(io, "Vertex Centered Stencil $((reverse(stencil.left)..., stencil.right...))")

struct InterfaceStencil{T, I, E}
    interior::NTuple{I, T}
    exterior::NTuple{E, T}
    edge::T
end

Base.show(io::IO, stencil::InterfaceStencil) = print(io, "Interface Centered Stencil $((reverse(stencil.interior)..., stencil.exterior..., stencil.edge))")

#############################
## Operators ################
#############################

"""
An abstract numerical operator, which implements stencils for vertices and subcells.
"""
abstract type Operator{T} end

"""
Builds the cell stencil for a given operator.
"""
cell_stencil(::Operator) = error("Unimplemented")

"""
Builds the left subcell stencil for a given operator.
"""
cell_left_stencil(::Operator) = error("Unimplemented")

"""
Builds the right subcell stencil for a given operator.
"""
cell_right_stencil(::Operator) = error("Unimplemented")

"""
Builds the vertex stencil for a given operator.
"""
vertex_stencil(::Operator) = error("Unimplemented")

interface_value_stencil(::Operator, L::Int, side::Bool) = error("Unimplemented")
interface_derivative_stencil(::Operator, L::Int, side::Bool) = error("Unimplemented")

############################
## Generic #################
############################

"""
Generates a cell centered grid of size `L + R + 1`
"""
cell_centered_grid(L, R) = ntuple(L + R + 1) do i
    i - L - 1//1
end

"""
Generates a vertex centered grid of size `L + R`
"""
vertex_centered_grid(L, R) = ntuple(L + R) do i
    i - L - 1//2
end

############################
## Lagrange stencils #######
############################

"""
A Lagrange operator. A generic class of `Operator` that utilizes lagrange basis polynomials to approximate function values
and derivatives.
"""
abstract type LagrangeOperator{T, O} <: Operator{T} end

function interface_value_stencil(::LagrangeOperator{T, O}, E::Int, side::Bool) where {T, O}
    L = ifelse(side, 2O, E)
    R = ifelse(side, E, 2O)

    grid = vertex_centered_grid(L, R)
    stencil = lagrange(grid, 0//1)

    left = reverse(ntuple(i -> T(stencil[i]), L))
    right = ntuple(i -> T(stencil[L + i]), R)

    if side
        interior = left
        exterior = ntuple(i -> right[i], R - 1)
        edge = right[end]
        return InterfaceStencil(interior, exterior, edge)
    else
        interior = right
        exterior = ntuple(i -> left[i], L - 1)
        edge = left[end]
        return InterfaceStencil(interior, exterior, edge)
    end
end

function interface_derivative_stencil(::LagrangeOperator{T, O}, E::Int, side::Bool) where {T, O}
    L = ifelse(side, 2O, E)
    R = ifelse(side, E, 2O)

    grid = vertex_centered_grid(L, R)
    stencil = lagrange_derivative(grid, 0//1)

    left = reverse(ntuple(i -> T(stencil[i]), L))
    right = ntuple(i -> T(stencil[L + i]), R)

    if side
        interior = left
        exterior = ntuple(i -> right[i], R - 1)
        edge = right[end]
        return InterfaceStencil(interior, exterior, edge)
    else
        interior = right
        exterior = ntuple(i -> left[i], L - 1)
        edge = left[end]
        return InterfaceStencil(interior, exterior, edge)
    end
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

"""
Stencils interpolating value using lagrange basis functions.
"""
struct LagrangeValue{T, O} <: LagrangeOperator{T, O} end

function cell_stencil(::LagrangeValue{T}) where T
    CellStencil(NTuple{0, T}(), one(T), NTuple{0, T}())
end

function cell_left_stencil(::LagrangeValue{T, O}) where {T, O}
    grid = cell_centered_grid(O, O)
    stencil = map(T, lagrange(grid, -1//2))
    lagrange_cell_stencil(stencil)
end

function cell_right_stencil(::LagrangeValue{T, O}) where {T, O}
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

"""
Stencils approximating derivatives using lagrange basis functions.
"""
struct LagrangeDerivative{T, O} <: LagrangeOperator{T, O} end

function cell_stencil(::LagrangeDerivative{T, O}) where {T, O}
    grid = cell_centered_grid(O, O)
    stencil = map(T, lagrange_derivative(grid, 0//1))
    lagrange_cell_stencil(stencil)
end

function cell_left_stencil(::LagrangeDerivative{T, O}) where {T, O}
    grid = cell_centered_grid(O, O)
    stencil = map(T, lagrange_derivative(grid, -1//2))
    lagrange_cell_stencil(stencil)
end

function cell_right_stencil(::LagrangeDerivative{T, O}) where {T, O}
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

"""
Stencils approximating second derivatives using lagrange basis functions.
"""
struct LagrangeDerivative2{T, O} <: LagrangeOperator{T, O} end

function cell_stencil(::LagrangeDerivative2{T, O}) where {T, O}
    grid = cell_centered_grid(O, O)
    stencil = map(T, lagrange_derivative_2(grid, 0//1))
    lagrange_cell_stencil(stencil)
end

function cell_left_stencil(::LagrangeDerivative2{T, O}) where {T, O}
    grid = cell_centered_grid(O, O)
    stencil = map(T, lagrange_derivative_2(grid, -1//2))
    lagrange_cell_stencil(stencil)
end

function cell_right_stencil(::LagrangeDerivative2{T, O}) where {T, O}
    grid = cell_centered_grid(O, O)
    stencil = map(T, lagrange_derivative_2(grid, 1//2))
    lagrange_cell_stencil(stencil)
end

function vertex_stencil(::LagrangeDerivative2{T, O}) where {T, O}
    grid = vertex_centered_grid(O + 1, O + 1)
    stencil = map(T, lagrange_derivative_2(grid, 0//1))
    lagrange_vertex_stencil(stencil)
end