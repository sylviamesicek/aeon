export DoFManager, nodefield

struct DoFManager{N, T}
    total::Int
    # A vector storing offsets for active cells and 0 for everything else
    offsets::Vector{Int}
    # A vector of active cells
    active::Vector{Int}
    # A vector of levels for the active cells
    levels::Vector{Int}
    # Depth of the mesh
    depth::Int

    function DoFManager(tree::TreeMesh{N, T}, depth::Int) where {N, T}
        # Build active vector
        active = []
        levels = []

        allactive(tree, depth) do node, level
            push!(active, node)
            push!(levels, level)
        end

        # Compute dof total and offsets
        total = 0
        offsets = zeros(length(tree))

        for a in active
            offsets[a] = total
            total += prod(nodedims(tree, a))
        end

        new{N, T}(total, offsets, active, levels, depth)
    end
end

DoFManager(tree::TreeMesh{N, T}) where {N, T} = DoFManager(tree, tree.maxdepth)

function nodefield(mesh::TreeMesh{N, T}, dofs::DoFManager{N, T}, node::Int, field::AbstractVector{T}) where {N, T} 
    dims = nodedims(mesh, node)
    view = @view field[(1:prod(dims)) .+ dofs.offsets[node]]
    reshape(view, dims)
end