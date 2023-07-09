############################
## Exports #################
############################

export Block, blockcells, blockbounds, blocktransform
export cellindices, cellwidths, cellcenter
export Field, evaluate, evaluate_point, interface
export Interface, evaluate_interface_interior, evaluate_interface_exterior

############################
## Block ###################
############################

"""
An abstract domain on which an operator may be applied
"""
abstract type Block{N, T} end

blockcells(::Block) = error("Unimplemented.")
blockbounds(::Block) = error("Unimplemented.")

function blocktransform(block::Block)
    bounds = blockbounds(block)
    Translate(bounds.origin) ∘ ScaleTransform(bounds.widths)
end

cellindices(block::Block) = CartesianIndices(blockcells(block))
cellwidths(block::Block{N, T}) where {N, T} = SVector{N, T}(1 ./ blockcells(block))
cellcenter(block::Block{N, T}, cell::CartesianIndex{N}) where {N, T} = SVector{N, T}((cell.I .- T(1//2)) ./ blockcells(block))

"""
A field defined over a domain (including the current block)
"""
abstract type Field{N, T} end

############################
## Point ###################
############################

"""
Converts a cell index to a point index
"""
cell_to_point(cell::Int) = 4cell - 1
cell_to_point(cell::CartesianIndex) = CartesianIndex(cell_to_point.(cell.I))

"""
Converts a point index to a cell index. If the point is not directly on a cell, this rounds down to the nearest cell to the left.
"""
point_to_cell(point) = (point + 1) ÷ 4
point_to_cell(point::CartesianIndex) = CartesianIndex(point_to_cell.(point.I))

"""
Computes the total number of points in a grid given the number of cells
"""
cell_total_to_points(total::Int) = 4total + 1
cell_total_to_points(total::NTuple{N, Int}) where N = cell_total_to_points.(total)

"""
Snaps a point to the given cell along an axis. 
"""
snap_point_on_axis(point::CartesianIndex, cell::Int, axis::Int) = CartesianIndex(setindex(point.I, cell_to_point(cell), axis))

"""
Tests if a point is a cell.
"""
is_point_cell(point::Int) = point % 4 == 3

"""
Tests if a point is a left subcell.
"""
is_point_cell_left(point::Int) = point % 4 == 2

"""
Tests if a point is a right subcell.
"""
is_point_cell_right(point::Int) = point % 4 == 0

"""
Tests if a point is a vertex.
"""
is_point_vertex(point::Int) = point % 4 == 1

############################
## Interface ###############
############################

struct Interface{T, F, E} 
    coefficients::NTuple{E, T}

    Interface{F}(coefs::NTuple{E, T}) where {T, F, E} = new{T, F, E}(coefs)
end

function evaluate_interface_interior(::Val{S}, point::CartesianIndex{N}, block::Block{N, T}, field::Field{N, T}, stencil::InterfaceStencil{T, I, E}, opers::NTuple{L, Operator{T}}) where{N, T, I, E, S, L}
    total_cells::Int = blockcells(block)[L]

    remaining = ntuple(i -> opers[i], Val(L - 1))

    if S
        result = zero(T)
        # Right side
        for i in 1:I
            offpoint = snap_point_on_axis(point, total_cells + 1 - i, L)
            value = evaluate_point(offpoint, block, field, remaining)
            result += stencil.interior[i] * value
        end

        return result
    else
        result = zero(T)
        # Left side
        for i in 1:I
            offpoint = snap_point_on_axis(point, i, L)
            value = evaluate_point(offpoint, block, field, remaining)
            result += stencil.interior[i] * value
        end

        return result
    end
end

function evaluate_interface_exterior(stencil::InterfaceStencil{T, I, E}, unknowns::NTuple{O, T}, i::Int) where{T, I, E, O}
    result = zero(T)
    for j in 1:i
        result += stencil.exterior[i] * unknowns[i]
    end
    result
end

############################
## Implements ##############
############################

"""
Evaluates the value of a field at a cell in a block. Must be implemented per block type.
"""
evaluate(cell::CartesianIndex{N}, block::Block{N, T}, field::Field{N, T}) where {N, T} = error("Evaluation unimplemented")

"""
Extends a centered stencil out of an interface of a block.
"""
interface(::Val{S}, point::CartesianIndex{N}, block::Block{N, T}, field::Field{N, T}, coefs::NTuple{O, T}, extent::Int, opers::NTuple{L, Operator{T}}) where {N, T, O, S, L} = error("Interface evaluation unimplemented.")


############################
## Evaluation ##############
############################

"""
Evaluates the action of a tensor product of operators on a field in a block at a cell.
"""
function evaluate(cell::CartesianIndex{N}, block::Block{N, T}, operators::NTuple{N, Operator{T}}, field::Field{N, T}) where {N, T}
    evaluate_point(cell_to_point(cell), block, operators, field)
end

"""
Evaluates the action of a tensor product of operators on a field in a block at an arbitray point.
"""
function evaluate_point(point::CartesianIndex{N}, block::Block{N, T}, operators::NTuple{N, Operator{T}}, field::Field{N, T}) where {N, T}
    evaluate_point(point, block, field, operators)
end

"""
Recursive function for evaluating the tensor product of a stencil at a point
"""
function evaluate_point(point::CartesianIndex{N}, block::Block{N, T}, field::Field{N, T}, opers::NTuple{L, Operator{T}}) where {N, T, L}
    if L == 0
        return evaluate(point_to_cell(point), block, field)
    end

    index = point[L]

    # Call the appropriate evaluate function depending on the type of the stencil
    if is_point_cell(index)
        return evaluate_cell(point, block, field, cell_stencil(opers[L]), opers)
    elseif is_point_vertex(index)
        return evaluate_vertex(point, block, field, vertex_stencil(opers[L]), opers)
    elseif is_point_cell_left(index)
        return evaluate_cell(point, block, field, cell_left_stencil(opers[L]), opers)
    else
        return evaluate_cell(point, block, field, cell_right_stencil(opers[L]), opers)
    end
end

#########################
## Helper ###############
#########################

function evaluate_cell(point::CartesianIndex{N}, block::Block{N, T}, field::Field{N, T}, stencil::CellStencil{T, O}, opers::NTuple{L, Operator{T}}) where {N, T, O, L}
    index = point[L]
    # Total number of cells
    total_cells::Int = blockcells(block)[L]
    central_cell::Int = point_to_cell(index)

    remaining = ntuple(i -> opers[i], Val(L - 1))
    
    result = stencil.center * evaluate_point(point, block, field, remaining)

    # Number of cells on left
    left_cells::Int = min(O, central_cell - 1)
    for off in 1:left_cells
        offpoint = snap_point_on_axis(point, central_cell - off, L)
        result += stencil.left[off] * evaluate_point(offpoint, block, field, remaining)
    end

    # Number of cells on right
    right_cells::Int = min(O, total_cells - central_cell)
    for off in 1:right_cells
        offpoint = snap_point_on_axis(point, central_cell + off, L)
        result += stencil.right[off] * evaluate_point(offpoint, block, field, remaining)
    end

    if left_cells < O
        result += interface(Val(false), point, block, field, stencil.left, left_cells, opers)  
    end

    if right_cells < O
        result += interface(Val(true), point, block, field, stencil.right, right_cells, opers)  
    end

    return result
end

function evaluate_vertex(point::CartesianIndex{N}, block::Block{N, T}, field::Field{N, T}, stencil::VertexStencil{T, O}, opers::NTuple{L, Operator{T}}) where {N, T, O, L}
    index = point[L]
    # Total number of cells
    total_cells::Int = blockcells(block)[L]
    # Cell to the left of the current point
    central_cell::Int = point_to_cell(index)

    remaining = ntuple(i -> opers[i], Val(L - 1))

    result = zero(T)

    left_cells::Int = min(O, central_cell)
    for off in 1:left_cells
        offpoint = snap_point_on_axis(point, central_cell - off + 1, L)
        result += stencil.left[off] * evaluate_point(offpoint, block, field, remaining)
    end

    right_cells::Int = min(O, total_cells - central_cell)
    for off in 1:right_cells
        offpoint = snap_point_on_axis(point, central_cell + off, L)
        result += stencil.right[off] * evaluate_point(offpoint, block, field, remaining)
    end

    if left_cells < O
        result += interface(Val(false), point, block, field, stencil.left, left_cells, opers)  
    end

    if right_cells < O
        result += interface(Val(true), point, block, field, stencil.right, right_cells, opers)  
    end

    return result
end
