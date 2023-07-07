############################
## Exports #################
############################

export Block, blockcells, blockbounds, blocktransform
export cellindices, cellwidths, cellcenter
export Field, evaluate, evaluate_point, interface
export Interface, evaluate_interface_interior

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

function evaluate_interface_interior(point::CartesianIndex{N}, block::Block{N, T}, field::Field{N, T}, face::FaceIndex{N}, stencil::InterfaceStencil{T, I, E}, rest::Operator{T}...) where{N, T, I, E}
    @debug @assert extent ≤ E

    axis = faceaxis(face)
    side = faceside(face)

    total_cells = blockcells(block)[axis]

    if side
        result = zero(T)
        # Right side
        for i in 1:I
            offpoint = snap_point_on_axis(point, total_cells + 1 - i, axis)
            result += stencil.interior[i] * evaluate_point(offpoint, block, field, rest...)
        end

        return result
    else
        result = zero(T)
        # Left side
        for i in 1:I
            offpoint = snap_point_on_axis(point, i, axis)
            result += stencil.interior[i] * evaluate_point(offpoint, block, field, rest...)
        end

        return result
    end
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
interface(point::CartesianIndex{N}, block::Block{N, T}, field::Field{N, T}, interface::Interface{N, T}, operator::Operator{T}, rest::Operator{T}...) where {N, T} = error("Interface evaluation unimplemented.")


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
Evaluates the value of a field in a block at a certain point. This point must be on a cell.
"""
function evaluate_point(point::CartesianIndex{N}, block::Block{N, T}, field::Field{N, T}) where {N, T}
    @debug @assert is_point_cell(point)
    evaluate(point_to_cell(point), block, field)
end

"""
Evaluates the action of a tensor product of operators on a field in a block at an arbitray point.
"""
function evaluate_point(point::CartesianIndex{N}, block::Block{N, T}, operators::NTuple{N, Operator{T}}, field::Field{N, T}) where {N, T}
    evaluate_point(point, block, field, operators...)
end

"""
Recursive function for evaluating the tensor product of a stencil at a point
"""
function evaluate_point(point::CartesianIndex{N}, block::Block{N, T}, field::Field{N, T}, operator::Operator{T}, rest::Operator{T}...) where {N, T}
    axis = N - length(rest)
    index = point[axis]
    
    # Call the appropriate evaluate function depending on the type of the stencil
    if is_point_vertex(index)
        stencil = vertex_stencil(operator)
        return evaluate_vertex(point, block, field, stencil,operator, rest...)
    elseif is_point_cell(index)
        stencil = cell_stencil(operator)
        return evaluate_cell(point, block, field, stencil, operator, rest...)
    elseif is_point_cell_left(index)
        stencil = cell_left_stencil(operator)
        return evaluate_cell(point, block, field, stencil, operator, rest...)
    else
        stencil = cell_right_stencil(operator)
        return evaluate_cell(point, block, field, stencil, operator, rest...)
    end
end

#########################
## Helper ###############
#########################

function evaluate_cell(point::CartesianIndex{N}, block::Block{N, T}, field::Field{N, T}, stencil::CellStencil{T, O}, operator::Operator{T}, rest::Operator{T}...) where {N, T, O}
    axis = N - length(rest)
    index = point[axis]
    # Total number of cells
    total_cells = blockcells(block)[axis]
    central_cell = point_to_cell(index)

    result = stencil.center * evaluate_point(point, block, field, rest...)

    # Number of cells on left
    left_cells = min(O, central_cell - 1)
    for off in 1:left_cells
        offpoint = snap_point_on_axis(point, central_cell - off, axis)
        result += stencil.left[off] * evaluate_point(offpoint, block, field, rest...)
    end

    # Number of cells on right
    right_cells = min(O, total_cells - central_cell)
    for off in 1:right_cells
        offpoint = snap_point_on_axis(point, central_cell + off, axis)
        result += stencil.right[off] * evaluate_point(offpoint, block, field, rest...)
    end

    L = O - left_cells
    R = O - right_cells

    if L > 0
        face = FaceIndex{N}(axis, false)
        coefs = ntuple(i -> stencil.left[left_cells + i], L)
        result += interface(point, block, field, Interface{face}(coefs), operator, rest...)    
    end

    if R > 0
        face = FaceIndex{N}(axis, true)
        coefs = ntuple(i -> stencil.right[right_cells + i], R)
        result += interface(point, block, field, Interface{face}(coefs), operator, rest...)    
    end

    return result
end

function evaluate_vertex(point::CartesianIndex{N}, block::Block{N, T}, field::Field{N, T}, stencil::VertexStencil{T, O}, operator::Operator{T}, rest::Operator{T}...) where {N, T, O}
    axis = N - length(rest)
    index = point[axis]
    # Total number of cells
    total_cells = blockcells(block)[axis]
    # Cell to the left of the current point
    central_cell = point_to_cell(index)

    result = zero(T)

    left_cells = min(O, central_cell)
    for off in 1:left_cells
        offpoint = snap_point_on_axis(point, central_cell - off + 1, axis)
        result += stencil.left[off] * evaluate_point(offpoint, block, field, rest...)
    end

    right_cells = min(O, total_cells - central_cell)
    for off in 1:right_cells
        offpoint = snap_point_on_axis(point, central_cell + off, axis)
        result += stencil.right[off] * evaluate_point(offpoint, block, field, rest...)
    end

    L = O - left_cells
    R = O - right_cells

    if L > 0
        face = FaceIndex{N}(axis, false)
        coefs = ntuple(i -> stencil.left[left_cells + i], L)
        result += interface(point, block, field, Interface{face}(coefs), operator, rest...)    
    end

    if R > 0
        face = FaceIndex{N}(axis, true)
        coefs = ntuple(i -> stencil.right[right_cells + i], R)
        result += interface(point, block, field, Interface{face}(coefs), operator, rest...)    
    end

    return result
end
