##############################
## Transfer ##################
##############################

export transfer_to_block!

function transfer_to_block!(f::F, block::AbstractBlock{N, T, O}, basis::AbstractBasis{T}, mesh::Mesh{N, T}, dofs::DoFManager{N, T}, level::Int, node::Int, values::AbstractVector{T}) where {N, T, O, F <: Function}
    cells = blockcells(block)
    dofs_per_block = prod(cells)

    offset = nodeoffset(dofs, level, node)
    transform = nodetransform(mesh, level, node)
    neighbors = nodeneighbors(mesh, level, node)

    # Fill Interior
    fill_interior_from_linear!(block) do i
        values[offset + i]
    end

    block_boundary_conditions!(block, basis, transform) do boundary, axis
        edge = boundary_edge(boundary, axis)
        face = FaceIndex{N}(axis, edge > 0)
        neighbor = neighbors[face]

        # If on boundary, use provided boundary conditions
        if neighbor < 0
            return f(boundary, axis)
        end

        # Otherwise we are using some manner of diritchlet boundary conditions to enforce
        # continuity between nodes.

        if neighbor == 0
            # Neighbor is coarser
            return diritchlet(T(1))
        elseif nodechildren(mesh, level, neighbor) == -1
            # Neighbor is same level
            neighbor_offset = nodeoffset(dofs, level, neighbor)

            offset_view = @view values[(1:dofs_per_block) .+ neighbor_offset]
            neighbor_view = ViewBlock{0}(reshape(offset_view, cells...))

            point = _boundary_to_point(block, boundary)
            neighbor_point = _boundary_to_reverse_point(block, boundary, axis)

            value = block_interior_prolong(block, Val(O), point, basis)
            neighbor_value = block_interior_prolong(neighbor_view, Val(O), neighbor_point, basis)

            return diritchlet(T(1), (value + neighbor_value)/2)
            # return diritchlet(T(1), value)
        else
            # Neighbor is more refined
            return diritchlet(T(1))
        end
    end
end

function _boundary_to_point(block::AbstractBlock{N}, boundary::BoundaryCell{N, I}) where {N, I}
    cells = blockcells(block)
    
    ntuple(Val(N)) do i
        if I[i] == 1
            VertexIndex(cells[i] + 1)
        elseif I[i] == -1
            VertexIndex(1)
        else
            CellIndex(boundary.cell[i])
        end
    end
end

function _boundary_to_reverse_point(block::AbstractBlock{N}, boundary::BoundaryCell{N, I}, axis::Int) where {N, I}
    cells = blockcells(block)

    ntuple(Val(N)) do i
        if I[i] == 1
            ifelse(i == axis, VertexIndex(1), VertexIndex(cells[i] + 1))
        elseif I[i] == -1
            ifelse(i == axis, VertexIndex(cells[i] + 1), VertexIndex(1))
        else
            CellIndex(boundary.cell[i])
        end
    end
end