export TreeMesh, allnodes, allactive, allleaves
export TreeNode, nodedims, nodepoints, nodetransform
export pointposition, pointgradient, pointhessian, pointvalue
export facereverse, faceside, faceaxis
export refine!, refineglobal!
export split_cart_to_linear, split_linear_to_cart

mutable struct TreeMesh{N, T, F}
    # Basic bounding info
    bounds::Vector{HyperBox{N, T}}
    # Tree structure
    parents::Vector{Int}
    children::Vector{Int}
    # Cached info
    # Stores neighbors as 1...N: left faces, then N+1:2N: right faces.
    # If a neighbor is 0, this indicates that the neighbor of this cell is coarser, and that one should recurse to
    # the parent to dertermine the neighbor on this face. If neighbor is less than zero, it indicates a boundary.
    neighbors::Vector{NTuple{F, Int}}
    # Stores max depth in this tree
    maxdepth::Int
    # Refinement level of whole tree
    refinement::Int

    function TreeMesh(rootbounds::HyperBox{N, T}, refinement::Int) where {N, T}
        bounds = [rootbounds]
        parents = [0]
        children = [0]
        neighbors = [ntuple(i -> -1, Val(2*N))]

        new{N, T, 2*N}(bounds, parents, children, neighbors, 1, refinement)
    end
end

############################
## Nodes ###################
############################

nodedims(mesh::TreeMesh{N}, ::Int) where {N} = ntuple(i -> 2^mesh.refinement + 1, Val(N))

nodetransform(mesh::TreeMesh, node::Int) = Translate(mesh.bounds[node].origin) ∘ ScaleTransform(mesh.bounds[node].widths)

nodepoints(mesh::TreeMesh, node::Int)  = CartesianIndices(nodedims(mesh, node))

############################
## Points ##################
############################

pointposition(mesh::TreeMesh{N, T}, ::Int, point::CartesianIndex{N}) where {N, T} = SVector{N, T}((point.I .- 1) .// 2^mesh.refinement)

pointvalue(::TreeMesh{N}, ::Int, point::CartesianIndex{N}, func::AbstractArray{T, N}) where {N, T} = func[point]

pointgradient(mesh::TreeMesh{N}, ::Int, point::CartesianIndex{N}, grad::Gradient{N, T}, func::AbstractArray{T, N}) where {N, T} = evaluate(point, grad, func) ./ 2^mesh.refinement

pointhessian(mesh::TreeMesh{N}, ::Int, point::CartesianIndex{N}, hess::Hessian{N, T}, func::AbstractArray{T, N}) where {N, T} = evaluate(point, hess, func) ./ 4^mesh.refinement

############################
## Faces ###################
############################

facereverse(face::Int, ::Val{N}) where N = ifelse(face > N, face - N, face + N)
faceside(face::Int, ::Val{N}) where N = face > N
faceaxis(face::Int, ::Val{N}) where N = ifelse(face > N, face - N, face)

############################
## Tree Iteration ##########
############################

Base.length(tree::TreeMesh) = length(tree.bounds)
Base.eachindex(tree::TreeMesh) = eachindex(tree.bounds)

function allnodes(f::Function, tree::TreeMesh{N, T}) where {N, T}
    stack = [1]
    newstack = []

    depth = 1

    while true
        if isempty(stack)
            break
        end

        for node in stack
            # Call function
            recurse = f(node, depth)
            # If there are children, recurse
            if recurse && tree.children[node] > 0
                for linear in 1:2^N
                    push!(newstack, tree.children[node] + linear)
                end
            end
        end
        # Swap stacks
        stack, newstack = newstack, empty(stack)
        depth += 1
    end
end

function allactive(f::Function, tree::TreeMesh, maxdepth::Int)
    allnodes(tree) do node, depth
        f(node, depth)
        depth < maxdepth
    end
end

allleaves(f::Function, tree::TreeMesh) = allactive(tree, tree.maxdepth) do node, depth
    if tree.children[node] == 0
        f(node, depth)
    end
end

##############################
## Refinement ################
##############################

function refine!(f::Function, tree::TreeMesh)
    shouldrefine = Vector{Bool}(undef, length(tree))
    fill!(shouldrefine, false)

    allleaves(tree) do node, level
        shouldrefine[node] = f(node, level)
    end

    _refine!(tree, shouldrefine)
end

refineglobal!(tree::TreeMesh) = refine!((node, level) -> true, tree)

function _refine!(tree::TreeMesh{N, T, F}, shouldrefine::Vector{Bool}) where {N, T, F}
    @assert length(tree.bounds) == length(shouldrefine)

    # Ensure tree is balanced
    rebalance!(tree, shouldrefine)

    newmaxdepth = tree.maxdepth + 1

    # Add children to nodes which should be refined
    for parent in eachindex(shouldrefine)
        if !shouldrefine[parent]
            continue
        end

        # Update max depth
        tree.maxdepth = newmaxdepth

        # Half widths for computing bounds
        halfwidths = tree.bounds[parent].widths ./ 2

        # Update children offset
        tree.children[parent] = length(tree)

        # Build children with basic neighborhood information
        for linear in 1:2^N
            # Get cartesian index of current child
            cart = split_linear_to_cart(linear, Val(N))
            # Build bounds using the halfwidths
            bounds = HyperBox(tree.bounds[parent].origin .+ halfwidths .* cart, halfwidths)
            # Construct neighborhood
            neighbors = ntuple(Val(F)) do face
                # Decode face
                side = faceside(face, Val(N))
                axis = faceaxis(face, Val(N))
                # Check whether face is interior, exterior, or boundary
                if cart[axis] == !side
                    # Interior face
                    # Find opposite child within node
                    opposite = setindex(cart, side, axis)
                    return tree.children[parent] + split_cart_to_linear(opposite)
                elseif tree.neighbors[parent][face] < 0
                    # Propogate boundary information
                    return tree.neighbors[parent][face]
                else
                    # Default to zero (indicates neighbor is coarser, might be changed by following loop).
                    return 0
                end
            end

            push!(tree.bounds, bounds) # Compute bounds
            push!(tree.parents, parent) # Has current node as parent
            push!(tree.children, 0) # This new node is a leaf
            push!(tree.neighbors, neighbors) # Set neighbors to tuple constructed above
        end
    end

    # Update exterior neighbors 
    for node in eachindex(shouldrefine)
        if !shouldrefine[node]
            continue
        end

        # Get parent
        parent = tree.parents[node]

        if parent == 0
            # Root
            continue
        end

        # For each face
        for face in 1:F
            # Decode face
            side = faceside(face, Val(N))
            axis = faceaxis(face, Val(N))

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
                    tree.neighbors[node] = setindex(tree.neighbors[node], otherchild, face)
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
                        tree.neighbors[child] = setindex(tree.neighbors[child], otherchild, face)
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

            parent = tree.parents[node]

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

########################
## Misc ################
########################

split_cart_to_linear(cart::NTuple{N, Bool}) where N = sum(ntuple(i -> 2^(i - 1) * cart[i], Val(N))) + 1
split_linear_to_cart(linear::Int, ::Val{N}) where N = ntuple(i -> (UInt(linear - 1) & UInt(2^(i - 1))) > 0, Val(N))