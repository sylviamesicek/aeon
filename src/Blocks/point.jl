export PointIndex, CellIndex, SubCellIndex, VertexIndex
export point_to_cell, subcell_side

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