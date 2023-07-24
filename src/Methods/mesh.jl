#######################
## Mesh ###############
#######################

export Level, Mesh, nodecells

"""
A single level of a mesh.
"""
struct Level{N, T, F}
    # Bounds for each node on this level
    bounds::Vector{HyperBox{N, T}}
    # Parent index for each node on this level
    parents::Vector{Int}
    # Child index for each node on this level
    children::Vector{Int}
    # Neighbors of each node. A negative value indicates
    # a physical boundary, 0 indicates the neighbor is on a coarser
    # level, and a positive number indicates the neighbor is on the
    # same level
    neighbors::Vector{HyperFaces{N, Int, F}}
    # Refinement flags
    flags::Vector{Bool}
end

function Level(bounds::HyperBox{N, T}) where {N, T}
    neighbors = nfaces(f -> -1, Val(N))
    Level{N, T, 2N}([bounds], [0], [-1], [neighbors], [false])
end

Base.length(level::Level) = length(level.bounds)
Base.eachindex(level::Level) = eachindex(level.bounds)

"""
A `Mesh` based on a quadtree.
"""
struct Mesh{N, T, F} 
    levels::Vector{Level{N, T, F}}
    refinement::Int

    function Mesh(bounds::HyperBox{N, T}, refinement::Int) where {N, T}
        new{N, T, 2N}([Level(bounds)], refinement)
    end
end

function Base.show(io::IO, mesh::Mesh{N, T}) where {N, T}
    print(io, "Mesh{$(N), $(T)}\n")
    for level in eachindex(mesh)
        print(io, "  Level $(level):\n")
        for node in eachnode(mesh, level)
            print(io, "    Node $(node):\n")
            print(io, "      bounds: $(nodebounds(mesh, level, node))\n")
            print(io, "      parent: $(nodeparent(mesh, level, node))\n")
            print(io, "      children: $(nodechildren(mesh, level, node))\n")
            print(io, "      neighbors: $(nodeneighbors(mesh, level, node))\n")
            print(io, "      flag: $(nodeflag(mesh, level, node))\n")
        end
    end
end

@generated function nodecells(mesh::Mesh{N}) where N
    quote
        Base.@ntuple $N i -> 2^mesh.refinement
    end
end

Base.length(mesh::Mesh) = length(mesh.levels)
Base.eachindex(mesh::Mesh) = eachindex(mesh.levels)

function Base.foreach(f::Function, mesh::Mesh)
    for level in eachindex(mesh)
        for node in eachindex(mesh.levels[level])
            f(level, node)
        end
    end
end

###############################
## Node Access ################
###############################

export eachnode, nodebounds, nodeparent, nodechildren, nodeneighbors, nodeflag
export leafnodes, nodetransform, eachleafnode

"""
An iterator over every node index on a level.
"""
function eachnode(mesh::Mesh, level::Int)
    eachindex(mesh.levels[level])
end

"""
An iterator over every leaf node index on a level.
"""
function eachleafnode(mesh::Mesh, level::Int)
    Iterators.filter(eachindex(mesh.levels[level])) do node
        nodechildren(mesh, level, node) == -1
    end
end

"""
Returns the bounds of a node.
"""
nodebounds(mesh::Mesh, level::Int, node::Int) = mesh.levels[level].bounds[node]

"""
Returns the parent of a node.
"""
nodeparent(mesh::Mesh, level::Int, node::Int) = mesh.levels[level].parents[node]

"""
Returns the children of a node.
"""
nodechildren(mesh::Mesh, level::Int, node::Int) = mesh.levels[level].children[node]

"""
Returns the neighbors of a node.
"""
nodeneighbors(mesh::Mesh, level::Int, node::Int) = mesh.levels[level].neighbors[node]

"""
Returns whether a node is flagged for refinement.
"""
nodeflag(mesh::Mesh, level::Int, node::Int) = mesh.levels[level].flags[node]

"""
Returns a transform from blockspace to globalspace.
"""
function nodetransform(mesh::Mesh, level::Int, node::Int)
    bounds = nodebounds(mesh, level, node)
    Translate(bounds.origin) ∘ ScaleTransform(bounds.widths)
end

Blocks.cellindices(mesh::Mesh) = CartesianIndices(nodecells(mesh))
Blocks.cellwidths(mesh::Mesh{N, T}) where {N, T} = SVector{N, T}(1 ./ nodecells(mesh))
Blocks.cellposition(mesh::Mesh{N, T}, cell::CartesianIndex{N}) where {N, T} = SVector{N, T}((cell.I .- T(1//2)) ./ nodecells(mesh))


###############################
## Refinement #################
###############################

export mark_refine!, mark_refine_global!
export prepare_refinement!, execute_refinement!, prepare_and_execute_refinement!

function mark_refine!(mesh::Mesh, level::Int, node::Int)
    if mesh.levels[level].children[node] == -1
        mesh.levels[level].flags[node] = true
    end
end

function mark_refine_global!(mesh::Mesh)
    foreach(mesh) do level, node
        mark_refine!(mesh, level, node)
    end
end

function prepare_refinement!(mesh::Mesh{N}) where N
    smooth = false

    while !smooth
        smooth = true

        for level in eachindex(mesh)
            for node in eachnode(mesh, level)
                if !nodeflag(mesh, level, node)
                    continue
                end

                # This node is both marked for refinement and a leaf.

                parent = nodeparent(mesh, level, node)
                neighbors = nodeneighbors(mesh, level, node)

                if parent == 0
                    # Root
                    continue
                end

                for face in faceindices(Val(N))
                    # Neighbor is coarser and should also be marked for refinement
                    if neighbors[face] == 0
                        # Get neighbor from parent
                        neighbor = nodeneighbors(mesh, level - 1, parent)[face]
                        # If it needed refinement, we are no longer smooth
                        smooth &= nodeflag(mesh, level - 1, neighbor)
                        # Flag for refinement
                        mark_refine!(mesh, level - 1, neighbor)
                    end
                end
            end
        end
    end
end

function execute_refinement!(mesh::Mesh{N, T, F}) where {N, T, F}
    # Determine if we have to add a new level to the mesh
    for node in eachnode(mesh, length(mesh))
        if nodeflag(mesh, length(mesh), node)
            push!(mesh.levels, Level{N, T, F}([], [], [], [], []))
            break
        end
    end

    # Add children to each level
    for level in 1:length(mesh) - 1
        coarse = mesh.levels[level]
        refined = mesh.levels[level + 1]

        for parent in eachindex(coarse)
            if !coarse.flags[parent]
                continue
            end

            coarse.children[parent] = length(refined)

            for child in splitindices(Val(N))
                bounds = splitbox(coarse.bounds[parent], child)

                neighbors = nfaces(Val(N)) do face
                    side = faceside(face)
                    axis = faceaxis(face)

                    if child[axis] == !side
                        # Interior face
                        other = splitreverse(child, axis)
                        node = coarse.children[parent] + other.linear
                        return node
                    else
                        neighbor = coarse.neighbors[parent][face]

                        if neighbor < 0
                            # Propogate physical boundary
                            return -1
                        end

                        return 0
                    end
                end

                push!(refined.bounds, bounds)
                push!(refined.parents, parent)
                push!(refined.children, -1)
                push!(refined.neighbors, neighbors)
                push!(refined.flags, false)
            end
        end
    end

    # Update exterior neighbors
    for level in 1:length(mesh) - 1
        coarse = mesh.levels[level]
        refined = mesh.levels[level + 1]

        for node in eachindex(coarse)
            if !coarse.flags[node]
                continue
            end

            parent = coarse.parents[node]

            if parent == 0
                # Root
                continue
            end

            for face in faceindices(Val(N))
                side = faceside(face)
                axis = faceaxis(face)

                # If this node was refined and neighbor was coarse
                # and also refined, we must update this node's faces
                if coarse.neighbors[node][face] == 0
                    # Neighbor is on a coarser level
                    neighbor = nodeneighbors(mesh, level-1, parent)[face]
                    # Index of current node within parent
                    child = SplitIndex{N}(node - nodechildren(mesh, level-1, parent))
                    # Find child on opposite side of face
                    other = splitreverse(child, axis)
                    # Set neighbors
                    coarse.neighbors[node] = setindex(coarse.neighbors[node], nodechildren(mesh, level-1, neighbor) + other.linear, face)
                end

                # By this point, neighbor should not be 0
                neighbor = coarse.neighbors[node][face]

                if neighbor < 0
                    # Do not touch boundaries
                    continue
                end

                if coarse.children[neighbor] == -1
                    # Neighbor is a leaf, so current node's children must have faces set to 0
                    for child in splitindices(Val(N))
                        if child[axis] == side
                            childnode = coarse.children[node] + child.linear
                            refined.neighbors[childnode] = setindex(refined.neighbors[childnode], 0, face)
                        end
                    end
                else
                    # Neighbor has children, so we update children according
                    for child in splitindices(Val(N))
                        if child[axis] == side
                            childnode = coarse.children[node] + child.linear
                            othernode = coarse.children[neighbor] + splitreverse(child, axis).linear
                            refined.neighbors[childnode] = setindex(refined.neighbors[childnode], othernode, face)
                        end
                    end
                end
            end
        end
    end

    for level in eachindex(mesh)
        fill!(mesh.levels[level].flags, false)
    end
end

function prepare_and_execute_refinement!(mesh::Mesh)
    prepare_refinement!(mesh)
    execute_refinement!(mesh)
end