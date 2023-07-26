export PointIndex, CellIndex, SubCellIndex, VertexIndex
export point_to_cell, subcell_side

"""
An abstract point index on a structured block grid. This includes cells, vertices, and subcells within the grid.
"""
abstract type PointIndex end

"""
The index of a cell. An `NTuple{N, CellIndex}` can be identity mapped to a cell `CartesianIndex{N}`.
"""
struct CellIndex <: PointIndex
    inner::Int
end

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

"""
A vertex index.
"""
struct VertexIndex <: PointIndex
    inner::Int
end

vertex_side(v::VertexIndex) = v.inner > 1

"""
Maps a point index to a corresponding cell for stencil generation. 
"""
function point_to_cell end

point_to_cell(v::CellIndex) = v.inner
point_to_cell(v::SubCellIndex) = (v.inner + 1) ÷ 2
point_to_cell(v::VertexIndex) = v.inner - ifelse(v.inner > 1, 1, 0)

#######################
## Prolongation #######
#######################

export block_prolong

"""
Performs prolongation for a full domain.
"""
function block_prolong(block::AbstractBlock{N, T, O}, point::NTuple{N, PointIndex}, basis::AbstractBasis{T}) where {N, T, O}
    cell = CartesianIndex(map(point_to_cell, point))
    stencils = map(i -> _point_to_prolong_stencil(i, basis, Val(O)), point) 
    block_stencil_product(block, cell, stencils)
end

function _point_to_prolong_stencil(::CellIndex, basis::AbstractBasis{T}, ::Val{O}) where {T, O}
    Stencil(basis, CellValue{0, 0}())
end

function _point_to_prolong_stencil(index::VertexIndex, basis::AbstractBasis{T}, ::Val{O}) where {T, O}
    if vertex_side(index)
        return Stencil(basis, VertexValue{O + 1, O + 1, true}())
    else
        return Stencil(basis, VertexValue{O + 1, O + 1, false}())
    end
end

function _point_to_prolong_stencil(index::SubCellIndex, basis::AbstractBasis{T}, ::Val{O}) where {T, O}
    if subcell_side(index)
        return Stencil(basis, SubCellValue{O, O, true}())
    else
        return Stencil(basis, SubCellValue{O, O, false}())
    end
end

######################
## Interior Prolong ##
######################

export block_interior_prolong

"""
Performs prolongation within a block, to the given order.
"""
function block_interior_prolong(block::AbstractBlock{N, T}, ::Val{O},  point::NTuple{N, PointIndex}, basis::AbstractBasis{T}) where {N, T, O}
    cells = blockcells(block)
    cell = CartesianIndex(map(point_to_cell, point))
    stencils = ntuple(i ->  _point_to_prolong_stencil_interior(point[i], cells[i], basis, Val(O)), Val(N))
    block_stencil_product(block, cell, stencils)
end

function _point_to_prolong_stencil_interior(::CellIndex, ::Int, basis::AbstractBasis{T}, ::Val{O}) where {T, O}
    Stencil(basis, CellValue{0, 0}())
end

@generated function _point_to_prolong_stencil_interior(index::VertexIndex, total::Int, basis::AbstractBasis{T}, ::Val{O}) where {T, O}
    quote 
        leftcells = min($(O + 1), index.inner - 1)
        rightcells = min($(O + 1), total - index.inner + 1)

        # Left side
        if leftcells ≤ $O
            if leftcells == 0
                return Stencil(basis, VertexValue{O, $(2O + 1), true}())
            end

            Base.@nexprs $O i -> begin
                if leftcells == i
                    return Stencil(basis, VertexValue{i, $(2O + 1), false}())
                end
            end
        end

        # Right side
        if rightcells ≤ $O
            if rightcells == 0
                return Stencil(basis, VertexValue{$(2O + 1), 0, false}())
            end

            Base.@nexprs $O i -> begin
                if rightcells == i
                    return Stencil(basis, VertexValue{$(2O + 1), i, false}())
                end
            end
        end

        return Stencil(basis, VertexValue{$(O + 1), $(O + 1), false}())
    end
end

@generated function _point_to_prolong_stencil_interior(index::SubCellIndex, total::Int, basis::AbstractBasis{T}, ::Val{O}) where {T, O}
    side_expr = side -> quote 
        cindex = point_to_cell(index)

        leftcells = min($O, cindex - 1)
        rightcells = min($O, total - cindex)

        # Left side
        if leftcells < $O
            Base.@nexprs $O i -> begin
                if leftcells == i - 1
                    return subcell_value_stencil(basis, Val(i - 1), Val($(2O + 1)), Val($side))
                end
            end
        end

        # Right side
        if rightcells < $O
            Base.@nexprs $O i -> begin
                if rightcells == i - 1
                    return subcell_value_stencil(basis, Val($(2O + 1), Val(i - 1)), Val($side))
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