##############################
## Transfer ##################
##############################

export transfer_to_block!

function transfer_to_block!(f::F, block::AbstractBlock{N, T, O}, values::AbstractVector{T}, mesh::Mesh{N, T}, dofs::DoFManager{N, T}, basis::AbstractBasis{T}, level::Int, node::Int) where {N, T, O, F <: Function}
    cells = blockcells(block)
    dofs_per_block = prod(cells)

    offset = nodeoffset(dofs, level, node)
    transform = nodetransform(mesh, level, node)
    neighbors = nodeneighbors(mesh, level, node)

    # Fill Interior
    fill_interior_from_linear!(block) do i
        values[offset + i]
    end

    block_physical_boundaries!(block, basis, transform) do boundary, axis
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
            return diritchlet(T(1), T(0))
        elseif nodechildren(mesh, level, neighbor) == -1
            # Neighbor is same level
            # neighbor_offset = nodeoffset(dofs, level, neighbor)

            # offset_view = @view values[(1:dofs_per_block) .+ neighbor_offset]
            # neighbor_view = ViewBlock{0}(reshape(offset_view, cells...))

            # point = _boundary_to_point(block, boundary)
            # neighbor_point = _boundary_to_reverse_point(block, boundary, axis)

            # value = block_interior_prolong(block, Val(O), point, basis)
            # neighbor_value = block_interior_prolong(neighbor_view, Val(O), neighbor_point, basis)

            # return diritchlet(T(1), (value + neighbor_value)/2)
            return diritchlet(T(1), T(0))
        else
            # Neighbor is more refined
            return diritchlet(T(1), T(0))
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

###################################
## Restriction ####################
###################################

function restrict!(f::F, y::AbstractVector{T}, x::AbstractVector{T}, mesh::Mesh{N, T}, dofs::DoFManager{N, T}, blocks::BlockManager{N, T}, basis::AbstractBasis{T}, maxlevel::Int) where {N, T, F <: Function}
    if maxlevel == 1
        # This is root, there is no smaller level
        return
    end

    for level in 1:(maxlevel - 1)
        # All lower levels are identity mapped.
        for node in eachnode(mesh, level)
            if nodechildren(mesh, level, node) == -1
                offset = nodeoffset(dofs, level, node)
                block = blocks[level]

                for (i, _) in enumerate(cellindices(block))
                    y[offset + i] = x[offset + i]
                end

            end
        end
    end

    if maxlevel ≤ mesh.base + 1
        # Restrict to single parent
        for node in eachnode(mesh, maxlevel)
            parent = nodeparent(mesh, maxlevel, node)
            poffset = nodeoffset(dofs, maxlevel-1, parent)
            block = blocks[maxlevel]

            transfer_to_block!(f, block, x, mesh, dofs, basis, level, node)

            for (i, pcell) in enumerate(CartesianIndices(blockcells(block) ./ 2))
                vertex = map(VertexIndex, pcell.I .* 2)
                y[poffset + i] = block_prolong(block, vertex, basis)
            end
        end
        
    else
        # Restrict to subcell of parent
        error("Restriction to node parents is unimplemented")
    end
end