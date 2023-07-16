#####################
## Points ###########
#####################

export PointIndex, CellIndex, SubCellIndex, VertexIndex
export subcell_side, subcell_to_cell, vertex_to_cell

"""
An abstract point index on a structured block grid.
"""
abstract type PointIndex end

"""
The index of a cell.
"""
struct CellIndex <: PointIndex
    inner::Int
end

point_to_cell(v::CellIndex) = v.inner

"""
A subcell index (ie, the two subcells along each axis which subdivide a cell)
"""
struct SubCellIndex <: PointIndex
    inner::Int
end

"""
Computes the side of this subcells subdivision
"""
subcell_side(v::SubCellIndex) = v.inner % 2 == 1

point_to_cell(v::SubCellIndex) = (v.inner + 1) ÷ 2

"""
A vertex index.
"""
struct VertexIndex <: PointIndex
    inner::Int
end

point_to_cell(v::VertexIndex) = v.inner - 1

########################
## Block ###############
########################

export Block, blockcells, blockbounds, blocktransform
export cellindices, cellwidths, cellcenter
export Field, value, setvalue!

abstract type Block{N, T} end

"""
Returns the number of cells that make up this block (must be implemented for each block type).
"""
blockcells(::Block) = error("Unimplemented")

"""
Returns the bounds of this block (must be implemented for each block type).
"""
blockbounds(::Block) = error("Unimplemented") 

"""
Returns a transform from block space to global space.
"""
function blocktransform(block::Block)
    bounds = blockbounds(block)
    Translate(bounds.origin) ∘ ScaleTransform(bounds.widths)
end

"""
Iterates the indices of a given block.
"""
cellindices(block::Block) = CartesianIndices(blockcells(block))
cellwidths(block::Block{N, T}) where {N, T} = SVector{N, T}(1 ./ blockcells(block))
cellcenter(block::Block{N, T}, cell::CartesianIndex{N}) where {N, T} = SVector{N, T}((cell.I .- T(1//2)) ./ blockcells(block))

# Field

"""
An abstract field defined over a domain.
"""
abstract type Field{N, T} end

"""
Retrieves or computes the value of a field at a cell in a block (must be implemented for each field type).
"""
value(field::Field{N, T}, block::Block{N, T}, cell::CartesianIndex{N}) where {N, T} = error("Unimplemented")

"""
Sets the value of a field at a cell in a block (must be implemented for each field type).
"""
setvalue!(field::Field{N, T}, value::T, block::Block{N, T}, cell::CartesianIndex{N}) where {N, T} = error("Unimplemented")

#############################
## Stencil Application ######
#############################

export block_stencil_product

"""
Apply the tensor product of a set of stencils at a point on a block.
"""
function block_stencil_product(field::Field{N, T}, block::Block{N, T}, cell::CartesianIndex{N}, stencils::NTuple{N, Stencil{T}}) where {N, T}
    _block_stencil_product(field, block, cell, stencils)
end

function _block_stencil_product(field::Field{N, T}, block::Block{N, T}, cell::CartesianIndex{N}, ::NTuple{0, Stencil{T}}) where {N, T}
    value(field, block, cell)
end

function _block_stencil_product(field::Field{N, T}, block::Block{N, T}, cell::CartesianIndex{N}, stencils::NTuple{L, Stencil{T}}) where {N, T, L}
    remaining = ntuple(i -> stencils[i], Val(L - 1))

    result = stencil[L].center * _block_stencil_product(field, block, cell, remaining)

    for (i, left) in enumerate(stencil[L].left)
        offcell = CartesianIndex(setindex(cell, cell[L] - i, L))
        result += left * _block_stencil_product(field, block, offcell, remaining)
    end

    for (i, right) in enumerate(stencil[L].right)
        offcell = CartesianIndex(setindex(cell, cell[L] + i, L))
        result += right * _block_stencil_product(field, block, offcell, remaining)
    end

    result
end

##################################
## Prolongation ##################
##################################

export blockprolong

"""
Performs prolongation within a block, to the given order.
"""
function blockprolong(field::Field{N, T}, block::Block{N, T}, point::NTuple{N, PointIndex}, basis::AbstractBasis{T}, ::Val{O}) where {N, T, O}
    totals = blockcells(block)
    cell = CartesianIndex(map(point_to_cell, point))
    stencils = map(i -> _point_to_prolong_stencil(i, totals[i], basis, Val(O)), point) 
    block_stencil_product(field, block, cell, stencils)
end

function _point_to_prolong_stencil(index::CellIndex, ::Int, basis::AbstractBasis{T}, ::Val{O}) where {T, O}
    cell_value_stencil(basis, Val(0), Val(0))
end

@generated function _point_to_prolong_stencil(index::VertexIndex, total::Int, basis::AbstractBasis{T}, ::Val{O}) where {T, O}
    quote 
        cindex = point_to_cell(index)

        leftcells = min($O, cindex)
        rightcells = min($O, total - cindex)

        # Left side
        if leftcells < $O
            Base.@nexprs $O i -> begin
                if leftcells == i - 1
                    return vertex_value_stencil(basis, Val(i - 1), Val($(2O + 1)), Val(false))
                end
            end
        end

        # Right side
        if rightcells < $O
            Base.@nexprs $O i -> begin
                if rightcells == i - 1
                    return vertex_value_stencil(basis, Val($(2O + 1)), Val(i - 1), Val(false))
                end
            end
        end

        return vertex_value_stencil(basis, Val($(O + 1)), Val($(O + 1)), Val(false))
    end
end


@generated function _point_to_prolong_stencil(index::SubCellIndex, total::Int, basis::AbstractBasis{T}, ::Val{O}) where {T, O}
    side_expr = side -> quote 
        cindex = point_to_cell(index)

        leftcells = min($O, cindex - 1)
        rightcells = min($O, total - cindex)

        # Left side
        if leftcells < $O
            Base.@nexprs $O i -> begin
                if leftcells == i - 1
                    return subcell_value_stencil(basis, Val(i - 1), Val($(2O)), Val($side))
                end
            end
        end

        # Right side
        if rightcells < $O
            Base.@nexprs $O i -> begin
                if rightcells == i - 1
                    return subcell_value_stencil(basis, Val($(2O), Val(i - 1)), Val($side))
                end
            end
        end

        return subcell_value_stencil(basis, Val($(O)), Val($(O)), Val($side))
    end

    quote 
        if subcell_side(index)
            $(side_expr(true))
        else
            $(side_expr(false))
        end
    end
end