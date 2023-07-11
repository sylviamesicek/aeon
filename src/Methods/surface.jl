export TreeSurface

struct TreeSurface{N, T} 
    tree::TreeMesh{N, T}
    # Total number of dofs on surface
    total::Int
    # Map from active node to global node
    active::Vector{Int}
    # Map from global node to offset into field
    offsets::Vector{Int}
    # Map from global node to active node
    nodes::Vector{Int}
    # Depth of mesh
    depth::Int

    function TreeSurface(tree::TreeMesh{N, T}, depth::Int) where {N, T}
        # Build active node
        active = []

        allactive(tree, depth) do node
            push!(active, node)
        end

        # Compute dof total and offsets
        total = 0
        offsets = zeros(length(tree))
        nodes = zeros(length(tree))

        for (i, a) in enumerate(active)
            offsets[a] = total
            nodes[a] = i
            total += prod(nodecells(tree))
        end

        new{N, T}(tree, total, active, offsets, nodes, depth)
    end
end

TreeSurface(tree::TreeMesh{N, T}) where {N, T} = TreeSurface(tree, tree.maxdepth)