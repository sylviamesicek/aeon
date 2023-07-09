export TreeMesh, meshroot, nodecells
export allnodes, allactive, allleaves
export mark_refine!, mark_global_refine!, execute_refinement!, prepare_refinement!, prepare_and_execute_refinement!

mutable struct TreeMesh{N, T, F}
    # Bounding info for each node
    bounds::Vector{HyperBox{N, T}}
    # Tree structure
    parents::Vector{Int}
    children::Vector{Int}
    # Neighbors of each node
    neighbors::Vector{HyperFaces{N, Int, F}}
    # Depths for each node
    depths::Vector{Int}
    # Refinement flags
    refineflags::Vector{Bool}
    # Max depth
    maxdepth::Int
    # Refinement level
    refinement::Int

    function TreeMesh(bounds::HyperBox{N, T}, refinement::Int) where {N, T}
        new{N, T, 2*N}([bounds], [0], [0], [nfaces(f -> -1, Val(N))], [1], [false], 1, refinement)
    end
end

#################################
## Helpers ######################
#################################

meshroot(::TreeMesh) = 1

function nodecells(tree::TreeMesh{N}) where {N} 
    refinement = tree.refinement 
    ntuple(dim -> 2^refinement, Val(N))
end

Base.length(tree::TreeMesh) = length(tree.bounds)
Base.eachindex(tree::TreeMesh) = eachindex(tree.bounds)

#################################
## Tree Iterations ##############
#################################


function allnodes(f::Function, tree::TreeMesh{N, T}) where {N, T}
    stack = [1]
    newstack = []

    while true
        if isempty(stack)
            break
        end

        for node in stack
            # Call function
            recurse = f(node)
            # If there are children, recurse
            if recurse && tree.children[node] > 0
                for child in splitindices(Val(N))
                    push!(newstack, tree.children[node] + child.linear)
                end
            end
        end
        # Swap stacks
        stack, newstack = newstack, empty(stack)
    end
end

function allactive(f::Function, tree::TreeMesh, maxdepth::Int)
    allnodes(tree) do node
        if tree.children[node] == 0 || tree.depths[node] == maxdepth
            f(node)
        end
        
        tree.depths[node] < maxdepth
    end
end

allleaves(f::Function, tree::TreeMesh) = allactive(f, tree, tree.maxdepth)

#########################
## Refinement ###########
#########################

function mark_refine!(tree::TreeMesh, node::Int)
    if tree.children[node] == 0
        tree.refineflags[node] = true
    end
end

function mark_global_refine!(tree::TreeMesh)
    allleaves(tree) do leaf
        mark_refine!(tree, leaf)
    end
end

function prepare_and_execute_refinement!(tree::TreeMesh)
    prepare_refinement!(tree)
    execute_refinement!(tree)
end

function execute_refinement!(tree::TreeMesh{N}) where N
    newmaxdepth = tree.maxdepth + 1

    oldtree = length(tree.refineflags)

    for parent in 1:oldtree
        if !tree.refineflags[parent]
            continue
        end

        # Update max depth
        tree.maxdepth = newmaxdepth

        # Update children offset
        tree.children[parent] = length(tree)

        # Build children with basic neighborhood information
        for child in splitindices(Val(N))
            bounds = splitbox(tree.bounds[parent], child)

            neighbors = nfaces(Val(N)) do face
                side = faceside(face)
                axis = faceaxis(face)

                if child[axis] == !side
                    # Interior face
                    # Find opposite child within node
                    other = splitreverse(child, axis)
                    return tree.children[parent] + other.linear
                else
                    # Propogate neighbor information
                    return tree.neighbors[parent][face]
                end
            end

            push!(tree.bounds, bounds)
            push!(tree.parents, parent)
            push!(tree.children, 0)
            push!(tree.neighbors, neighbors)
            push!(tree.depths, tree.depths[parent] + 1)
            push!(tree.refineflags, false)
        end
    end

    # Update exterior neighbors
    for node in 1:oldtree
        if !tree.refineflags[node]
            continue
        end

        parent = tree.parents[node]

        if parent == 0
            # Root
            continue
        end

        # For each face
        for face in faceindices(Val(N))
            side = faceside(face)
            axis = faceaxis(axis)

            # If refined and neighbor was coarse, update node's faces
            if tree.neighbors[node][face] == 0
                # Neighbor is on coarser level
                neighbor = tree.neighbors[parent][face]
                # Index of current node within parent
                child = SplitIndex{N}(node - tree.children[parent])


                # I don't think this will ever be true
                if tree.children[neighbor] == 0 || child[axis] != side
                    continue
                end

                # Find child on oppose side of face
                other = splitreverse(child, axis)
                # Set neighbor
                tree.neighbors[node][face] = tree.children[neighbor] + other.linear
            end

            neighbor = tree.neighbors[node][face]

            # Check neighbor on this face
            if neighbor < 0
                # Do not touch boundaries
                continue
            elseif tree.children[neighbor] == 0
                # Neighbor is leaf, so current nodes children faces must be set to 0
                for child in splitindices(Val(N))
                    if child[axis] == side
                        childnode = tree.children[node] + child.linear

                        tree.neighbors[childnode][face] = 0
                    end
                end
            else
                # Neighbor is of equal or more refined level. Update accordinly
                for child in splitindices(Val(N))
                    # Filter for children touching this face.
                    if child[axis] == side
                        childnode = tree.children[node] + child.linear
                        othernode = tree.children[neighbor] + splitreverse(child, axis).linear
                        tree.neighbors[childnode][face] = othernode
                    end
                end
            end
        end
    end
end

function prepare_refinement!(tree::TreeMesh{N}) where {N}
    # Track smoothness
    smooth = false

    while !smooth
        smooth = true

        for node in eachindex(tree)
            if !tree.refineflags[node]
                continue
            end

            parent = tree.parents[node]

            if parent == 0
                # Root
                continue
            end

            for face in faceindices(Val(N))
                # Neighbor is on different level but must be leaf
                if tree.neighbors[node][face] == 0
                    # Get neighbor from parent
                    neighbor = tree.neighbors[parent][face]
                    # If it needed refinement, we are no longer smooth
                    smooth &= tree.refineflags[neighbor]
                    # Flag for refinement
                    tree.refineflags[neighbor] = true
                end
            end
        end
    end
end