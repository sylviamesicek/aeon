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

struct DoFLevel{N, T} 
    # Stores total number of DoFs on this level
    total::Vector{Int}
    # Stores offsets into dof vector for each cell
    offsets::Vector{Int}
    # A list of active cells in this level
    cells::Vector{Int}

    # Stores parent of each cell. If this is negative, it indicates that the cell is the same refinement
    # depth on the previous level, so no prolongation or restriction is needed.
    parents::Vector{Int}
    # Stores children of each cell. If this is negative it indicates that there is only one child, and it
    # is of the same refinement.
    children::Vector{Int}
end

struct DoFManager2{N, T} 
    levels::DoFLevel{Int}

    function DoFManager2(tree::TreeMesh{N, T}) where {N, T}
        root = meshroot(tree)
        roottotal = prod(nodedims(mesh, root))
        rootoffsets = [0]
        rootcells = [root]
        rootparents = [0]
        rootchildren = [0]

        levels = [DoFLevel{N, T}(roottotal, rootoffsets, rootcells, rootparents, rootchildren)]

        depth = 1

        while depth ≤ tree.maxdepth
            prev = levels[depth]

            cells = []
            parents = []
            children = []

            for (i, cell) in enumerate(prev.cells)
                if tree.children[cell] == 0
                    push!(cells, cell)
                    push!(parents, i)
                    push!(children, 0)
                end
            end

            total = 0
            offsets = []
            
            cells = copy(prev.cells)
            levels = copy(prev.levels)


            depth += 1
        end

    end
end