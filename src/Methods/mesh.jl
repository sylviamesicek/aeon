export Mesh, meshdims, meshcells
export hyperprism, refine!

"""
The overall topology of a domain. The mesh class
provides a means of discretizing, building, and manpulating numerical domains.
It can be iterated to yield the individual cells of the mesh. 
"""
struct Mesh{N, T}
    cells::Array{Cell{N, T}, N}
end

meshdims(mesh::Mesh) = size(mesh.cells)
meshcells(mesh::Mesh) = CartesianIndices(meshdims(mesh))

Base.getindex(mesh::Mesh{N}, index::CartesianIndex{N}) where {N} = mesh.cells[index]
Base.setindex!(mesh::Mesh{N}, cell::Cell, index::CartesianIndex{N}) where {N} = mesh.cells[index] = cell

function hyperprism(bounds::HyperBox{N, T}, meshdims::NTuple{N, Int}, refinement::Int) where {N, T}
    @assert N > 0

    # Array of final cells for the mesh.
    cells = Array{Cell{N, T}, N}(undef, celldims...)

    cellwidths = bounds.widths ./ meshdims

    for index in CartesianIndices(celldims)
        # Compute position of origin
        position = bounds.origin + SVector{N, T}((index.I .- 1) .* cellwidths)
        # Add cell to mesh
        cells[index] = Cell(HyperBox(position, cellwidths), refinement)
    end

    Mesh{N, T}(cells)
end

"""
Refine a mesh using a refinement vector. For each `true` in the `shouldrefine` vector, the number of dofs on the corresponding cell is doubled along each
axis.
"""
function refine!(mesh::Mesh{N}, shouldrefine::Array{Bool, N}) where N
    smooth = false
    while !smooth
        smooth = smoothrefine!(mesh, shouldrefine)
    end

    # We now have smoothed the shouldrefine vector. Recompute cells.

    for cell in eachindex(mesh)
        if shouldrefine[cell]
            c = mesh[cell]
            mesh[cell] = Cell(c.bounds, c.refinement + 1)
        end
    end
end

"""
Refines a mesh based on a functional predicate.
"""
function refine!(pred::Function, mesh::Mesh)
    shouldrefine = Array{Bool}(undef, size(mesh.cells)...)

    for cell in eachindex(shouldrefine)
        shouldrefine[cell] = pred(cell)
    end

    refine!(mesh, shouldrefine)
end

"""
Smooths the `shouldrefine` vector. Returns true if the vector is already smooth.
"""
function smoothrefine!(mesh::Mesh{N, T}, shouldrefine::Array{Bool, N}) where {N, T}
    smooth = true

    for cell in CartesianIndices(size(mesh.cells))
        if shouldrefine[cell]
            for dim in 1:N
                left = CartesianIndex(cell.I .- ntuple(i -> i == dim, Val(N)))

                if left[dim] > 0 && !shouldrefine[left] && mesh[left].refinement < mesh[cell].refinement
                    smooth = false
                    shouldrefine[left] = true
                end

                right = CartesianIndex(cell.I .+ ntuple(i -> i == dim, Val(N)))

                if right[dim] > 0 && !shouldrefine[right] && mesh[right].refinement < mesh[cell].refinement
                    smooth = false
                    shouldrefine[right] = true
                end
            end
        end
    end

    smooth
end

struct TreeMesh{N, T, F}
    bounds::Vector{HyperBox{N, T}}
    parents::Vector{Int}
    children::Vector{Int}
    # Stores neighbors as 1...N: left faces, then N+1:2N: right faces.
    # If a neighbor is 0, this indicates that the neighbor of this cell is coarser, and that one should recurse to
    # the parent to dertermine the neighbor on this face. If neighbor is less than zero, it indicates a boundary.
    neighbors::Vector{NTuple{F, Int}}
    # Stores level of a particular node
    levels::Vector{Int}
    refinement::Int

    function TreeMesh(rootbounds::HyperBox{N, T}, refinement::Int) where {N, T}
        bounds = [rootbounds]
        parents = [0]
        children = [0]
        neighbors = [ntuple(i -> -1, Val(2*N))]
        levels = [1]

        new{N, T, 2*N}(bounds, parents, children, neighbors, levels, refinement)
    end
end

function refine!(tree::TreeMesh{N, T, F}, shouldrefine::Vector{Bool}) where {N, T, F}
    @assert length(tree.bounds) == length(shouldrefine)

    # Only refine children
    shouldrefine .&= tree.children .== 0

    # Ensure tree is balanced
    rebalance!(tree, shouldrefine)

    # Add children to nodes which should be refined
    for node in eachindex(shouldrefine)
        if !shouldrefine[node]
            continue
        end

        # Half widths for computing bounds
        halfwidths = tree.bounds[node].widths ./ 2

        # Update children offset
        tree.children[node] = length(tree.bounds)

        # Get level of current node
        level = tree.levels[node]

        # Build children with basic neighborhood information
        for linear in 1:2^N
            # Get cartesian index of current child
            cart = split_linear_to_cart(linear, Val(N))
            # Build bounds using the halfwidths
            bounds = HyperBox(tree.bounds[node].origin .+ halfwidths .* cart, halfwidths)
            # Construct neighborhood
            neighbors = ntuple(Val(F)) do face
                # Decode face
                side = face > N
                axis = ifelse(side, face, face - N)
                # Check whether face is interior, exterior, or boundary
                if cart[axis] == !side
                    # Interior face
                    # Find opposite child within node
                    opposite = setindex(cart, side, axis)
                    return tree.children[node] + split_cart_to_linear(opposite)
                elseif tree.neighbors[face] < 0
                    # Propogate boundary information
                    return tree.neighbors[face]
                else
                    # Default to zero (indicates neighbor is coarser)
                    return 0
                end
            end

            push!(tree.bounds, bounds) # Compute bounds
            push!(tree.parents, node) # Has current node as parent
            push!(tree.children, 0) # This new node is a leaf
            push!(tree.neighbors, neighbors) # Set neighbors
            push!(tree.levels, level + 1) # One level more refined
        end
    end

    # Update exterior neighbors 
    for node in eachindex(shouldrefine)
        if !shouldrefine[node]
            continue
        end

        # Get parent
        parent = tree.parent[node]

        if parent == 0
            # Root
            continue
        end

        # For each face
        for face in 1:F
            # Decode face
            side = face > N
            axis = ifelse(side, face, face - N)

            # Check neighbor on this face
            if tree.neighbors[node][face] < 0
                # Do not tocuh boundaries
                continue
            elseif tree.neighbors[node][face] == 0
                # Neighbor is on coarser level
                neighbor = tree.neighbors[parent][face]
                # I don't think this will ever be true
                if !shouldrefine[neighbor]
                    continue
                end

                # Index of current node within parent
                linear = node - tree.children[parent]
                # Get cartesian index
                cart = split_linear_to_cart(linear, Val(N))
                # Filter to make sure node is actually touching this face
                if cart[axis] == side
                    # Find node on opposite side of face
                    othercart = setindex(Tuple(cart), !side, axis)
                    # Convert to linear indexing
                    otherlinear = split_cart_to_linear(othercart)
                    # Find child of other node
                    otherchild = tree.children[neighbor] + otherlinear
                    # Update current node's neighbors
                    tree.neighbors[node] = setindex(tree.neightbors[node], otherchild, face)
                end
            else
                # Neighbor is on equal or refined level
                neighbor = tree.neighbors[node][face]
                # Only update if necessary
                if !shouldrefine[neighbor]
                    continue
                end

                # For each child
                for linear in 1:2^N
                    # Get global index for child
                    child = tree.children[node] + linear
                    # Get cartesian index
                    cart = split_linear_to_cart(linear, Val(N))
                    # Filter for children touching this face
                    if cart[axis] == side
                        # Find node on opposite side of face
                        othercart = setindex(Tuple(cart), !side, axis)
                        # Convert to linear indexing
                        otherlinear = split_cart_to_linear(othercart)
                        # Find child of other node
                        otherchild = tree.children[neighbor] + otherlinear
                        # Update neighbors of child
                        tree.neighbors[child] = setindex(tree.neightbors[child], otherchild, face)
                    end
                end
            end
        end
    end
end


function rebalance!(tree::TreeMesh{N, T, F}, shouldrefine::Vector{Bool}) where {N, T, F}
    # Track smoothness
    smooth = false

    while !smooth
        smooth = true

        for node in eachindex(shouldrefine)
            if !shouldrefine[node]
                continue
            end

            parent = tree.parent[node]

            if parent == 0
                # Root
                continue
            end

            for face in 1:F
                # Neighbor is on different level
                if tree.neighbors[node][face] == 0
                    # Get neighbor from parent
                    neighbor = tree.neighbors[parent][face]
                    # If it needed refinement, we are no longer smooth
                    smooth &= shouldrefine[neighbor]
                    # Flag for refinement
                    shouldrefine[neighbor] = true
                end
            end
        end
    end
end

split_cart_to_linear(cart::NTuple{N, Bool}) where N = sum(ntuple(i -> 2^(i - 1) * cart[i], Val(N))) + 1
split_linear_to_cart(linear::Int, ::Val{N}) where N = ntuple(i -> (UInt(linear - 1) & UInt(2^(i - 1))) > 0, Val(N))