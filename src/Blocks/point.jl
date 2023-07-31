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
subcell_side(v::SubCellIndex) = v.inner % 2 == 0

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

##############################
## Prolongation Total ########
##############################

export block_prolong_total

function block_prolong_total(block::AbstractBlock{N, T, O}, point::NTuple{N, PointIndex}, basis::AbstractBasis{T}) where {N, T, O}
    cell = CartesianIndex(map(point_to_cell, point))
    stencils = prolong_total_stencils(block, point, basis)
    block_stencil_product(block, cell, stencils)
end

function prolong_total_stencils(::AbstractBlock{N, T, O}, point::NTuple{N, PointIndex}, basis::AbstractBasis{T}) where {N, T, O}
    map(p -> _prolong_total_stencil(p, basis, Val(O)), point)
end

function _prolong_total_stencil(::CellIndex, basis::AbstractBasis{T}, ::Val{O}) where {T, O}
    Stencil(basis, CellValue{0, 0}())
end

function _prolong_total_stencil(index::VertexIndex, basis::AbstractBasis{T}, ::Val{O}) where {T, O}
    if vertex_side(index)
        return Stencil(basis, VertexValue{O + 1, O + 1, true}())
    else
        return Stencil(basis, VertexValue{O + 1, O + 1, false}())
    end
end

function _prolong_total_stencil(index::SubCellIndex, basis::AbstractBasis{T}, ::Val{O}) where {T, O}
    if subcell_side(index)
        return Stencil(basis, SubCellValue{O, O, true}())
    else
        return Stencil(basis, SubCellValue{O, O, false}())
    end
end

#######################
## Prolongation #######
#######################

export block_prolong

function block_prolong(block::AbstractBlock{N, T, O}, point::NTuple{N, PointIndex}, basis::AbstractBasis{T}) where {N, T, O}
    cell = CartesianIndex(map(point_to_cell, point))
    stencils = prolong_stencils(block, point, basis)
    block_stencil_product(block, cell, stencils)
end

@generated function prolong_stencils(block::AbstractBlock{N, T, O}, point::NTuple{N, PointIndex}, basis::AbstractBasis{T}) where {N, T, O}
    quote
        cells = blockcells(block)
        Base.@ntuple $N i -> _prolong_stencil(point[i], cells[i], basis, Val(O))
    end
end

function _prolong_stencil(::CellIndex, ::Int, basis::AbstractBasis{T}, ::Val{O}) where {T, O}
    Stencil(basis, CellValue{0, 0}())
end

@generated function _prolong_stencil(index::VertexIndex, total::Int, basis::AbstractBasis{T}, ::Val{O}) where {T, O}
    quote 
        leftcells = min($(O + 1), index.inner - 1)
        rightcells = min($(O + 1), total - index.inner + 1)

        # Left side
        if leftcells ≤ $O
            if leftcells == 0
                return Stencil(basis, VertexValue{0, $(2O + 2), true}())
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
                return Stencil(basis, VertexValue{$(2O + 2), 0, false}())
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

@generated function _prolong_stencil(index::SubCellIndex, total::Int, basis::AbstractBasis{T}, ::Val{O}) where {T, O}
    side_expr = side -> quote 
        cindex = point_to_cell(index)

        leftcells = min($(O + 1), cindex - 1)
        rightcells = min($(O + 1), total - cindex)

        # Left side
        if leftcells < $(O + 1)
            Base.@nexprs $(O + 1) i -> begin
                if leftcells == i - 1
                    return Stencil(basis, SubCellValue{i - 1, $(2O + 1), $side}())
                end
            end
        end

        # Right side
        if rightcells < $(O + 1)
            Base.@nexprs $(O + 1) i -> begin
                if rightcells == i - 1
                    return Stencil(basis, SubCellValue{$(2O + 1), i - 1, $side}())
                end
            end
        end

        return Stencil(basis, SubCellValue{$(O + 1), $(O + 1), $side}())
    end

    quote 
        if subcell_side(index)
            $(side_expr(true))
        else
            $(side_expr(false))
        end
    end
end