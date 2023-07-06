############################
## Exports #################
############################

export Block, blockcells, blockbounds, blocktransform
export cellindices, cellwidths, cellcenter
export Field
export evaluate, interface, evaluate_point
export cell_to_point, point_to_cell

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

"""
Evaluates the value of a field at a cell in a block.
"""
evaluate(cell::CartesianIndex{N}, block::Block{N, T}, field::Field{N, T}) where {N, T} = error("Evaluation unimplemented")

"""
Extends a centered stencil out of an interface of a block.
"""
interface(point::CartesianIndex{N}, block::Block{N, T}, field::Field{N, T}, face::FaceIndex{N}, coefs::InterfaceCoefs{T}, operator::Operator{T}, rest::Operator{T}...) where {N, T} = error("Interface evaluation unimplemented.")

############################
## Point ###################
############################

cell_to_point(cell::Int) = 4cell - 1
cell_to_point(cell::CartesianIndex) = CartesianIndex(cell_to_point.(cell.I))

point_to_cell(point) = (point + 1) ÷ 4
point_to_cell(point::CartesianIndex) = CartesianIndex(point_to_cell.(point.I))

cell_total_to_points(total::Int) = 4total + 1

is_point_cell(point::Int) = point % 4 == 3
is_point_cell_left(point::Int) = point % 4 == 2
is_point_cell_right(point::Int) = point % 4 == 0
is_point_vertex(point::Int) = point % 4 == 1

############################
## Evaluation ##############
############################

function evaluate(cell::CartesianIndex{N}, block::Block{N, T}, operators::NTuple{N, Operator{T}}, field::Field{N, T}) where {N, T}
    evaluate_point(cell_to_point(cell), block, operators, field)
end

function evaluate_point(point::CartesianIndex{N}, block::Block{N, T}, field::Field{N, T}) where {N, T}
    evaluate(point_to_cell(point), block, field)
end

function evaluate_point(point::CartesianIndex{N}, block::Block{N, T}, operators::NTuple{N, Operator{T}}, field::Field{N, T}) where {N, T}
    evaluate_point(point, block, field, operators...)
end

"""
Recursive function for evaluating the tensor product of a stencil at a point
"""
function evaluate_point(point::CartesianIndex{N}, block::Block{N, T}, field::Field{N, T}, operator::Operator{T}, rest::Operator{T}...) where {N, T}
    axis = N - length(rest)
    index = point[axis]
    
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

function evaluate_cell(point::CartesianIndex{N}, block::Block{N, T}, field::Field{N, T}, stencil::CellStencil{T}, operator::Operator{T}, rest::Operator{T}...) where {N, T}
    axis = N - length(rest)
    index = point[axis]
    # Total number of cells
    ctotal = blockcells(block)[axis]
    # Cell either on, or to the left of the current point
    cindex = point_to_cell(index)

    @assert is_point_cell(index) "Point must be a cell index"

    O = order(stencil)
    result = stencil.center * evaluate_point(point, block, field, rest...)

    maxleft = min(O, cindex)
    for off in 1:maxleft
        offpoint = CartesianIndex(setindex(point.I, cell_to_point(cindex - off), axis))
        result += stencil.left[off] * evaluate_point(offpoint, block, field, rest...)
    end

    maxright = min(O, ctotal - cindex)
    for off in 1:maxright
        offpoint = CartesianIndex(setindex(point.I, cell_to_point(cindex + off), axis))
        result += stencil.cright[off] * evaluate_point(offpoint, block, field, rest...)
    end

    if maxleft < O
        face = FaceIndex{N}(axis, false)
        coefs = InterfaceCoefs(stencil, false, O - maxleft)
        result += interface(point, block, field, face, coefs, operator,  rest...)
    end

    if maxright < O
        face = FaceIndex{N}(axis, true)
        coefs = InterfaceCoefs(stencil, true, O - maxright)
        result += interface(point, block, field, face, coefs, operator, rest...)
    end

    return result
end


function evaluate_vertex(point::CartesianIndex{N}, block::Block{N, T}, field::Field{N, T}, stencil::VertexStencil{T}, operator::Operator{T}, rest::Operator{T}...) where {N, T}
    axis = N - length(rest)
    index = point[axis]
    # Total number of cells
    ctotal = blockcells(block)[axis]
    # Cell either on, or to the left of the current point
    cindex = point_to_cell(index)

    O = order(stencil)
    result = zero(T)

    maxleft = min(O, cindex)
    for off in 1:maxleft
        offpoint = CartesianIndex(setindex(point.I, cell_to_point(cindex - off + 1), axis))
        result += stencil.left[off] * evaluate_point(offpoint, block, field, rest...)
    end

    maxright = min(O, ctotal - cindex)
    for off in 1:maxright
        offpoint = CartesianIndex(setindex(point.I, cell_to_point(cindex + off), axis))
        result += stencil.cright[off] * evaluate_point(offpoint, block, field, rest...)
    end

    if maxleft < O
        face = FaceIndex{N}(axis, false)
        coefs = InterfaceCoefs(stencil, false, O - maxleft)
        result += interface(point, block, field, face, coefs, operator, rest...)
    end

    if maxright < O
        face = FaceIndex{N}(axis, true)
        coefs = InterfaceCoefs(stencil, true, O - maxright)
        result += interface(point, block, field, face, O - maxright, coefs, operator, rest...)
    end

    return result
end
