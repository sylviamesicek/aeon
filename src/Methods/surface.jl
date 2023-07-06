export TreeSurface, TreeField, TreeBlock

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

struct TreeField{N, T} <: Field{N, T}
    values::Vector{T}
end

Base.similar(surface::TreeSurface{N, T}) where {N, T} = TreeField{N, T}(Vector{T}(undef, surface.total))

struct TreeBlock{N, T} <: Block{N, T}
    surface::TreeSurface{N, T}
    node::Int
end

Operators.blockcells(block::TreeBlock) = nodecells(block.surface.tree)
 
function Operators.evaluate(point::CartesianIndex{N}, block::TreeBlock{N, T}, field::TreeField{N, T})
    # Find offset 
    offset = block.surface.offsets[block.node]
    # Build map to linear
    linear = LinearIndices(nodecells(block.surface.tree))
    # Get global ptr
    ptr = linear[point] + offset
    # Access value
    return field[ptr]
end

function Operators.interface(point::CartesianIndex{N}, block::TreeBlock{N, T}, field::TreeField{N, T}, face::FaceIndex{N}, coefficients::Coefficients{T, L, O}, rest::Stencil{T}...) where {N, T, L, O}
    surface = block.surface
    tree = surface.tree
    node = block.node

    axis = faceaxis(face)
    side = faceside(face)

    neighbor = tree.neighbors[node][face]

    total = nodecells(tree)[axis]

    if neighbor > 0 && surface.nodes[neighbor] > 0
        # There is an active neighbor on the same level.

        # Block to the left of the boundary
        leftblock = side ? block : TreeBlock(surface, neighbor)
        # Block to the right of the boundary
        rightblock = side ? TreeBlock(surface, neighbor) : block
        
        # Generate stencil matrix for right and left side
        valuestencilsleft = boundary_value_left(Val(T), Val(L), Val(O))
        valuestencilsright = boundary_value_right(Val(T), Val(L), Val(O))

        derivativestencilsleft = boundary_derivative_left(Val(T), Val(L), Val(O))
        derivativestencilsright = boundary_derivative_right(Val(T), Val(L), Val(O))

        # First index is closest to interface
        unknownleft = ntuple(i -> zero(T), Val(L))
        unknownright = ntuple(i -> zero(T), Val(L))
        
        for num in 1:L
            # Accumulate rhs values
            valuerhs = zero(T)
            derivativerhs = zero(T)

            # Stencils
            valuestencilleft = valuestencilsleft[num]
            valuestencilright = valuestencilsright[num]
            derivativestencilleft = derivativestencilsleft[num]
            derivativestencilright = derivativestencilsright[num]

            # For each value on their own domain
            for i in 1:O
                rightpoint = CartesianIndex(setindex(point.I, O + 1 - i, axis))
                rightvalue = evaluate(rightpoint, rightblock, field, rest...)
                valuerhs += valuestencilleft[end + 1 - i] * rightvalue
                derivativerhs += derivativestencilleft[end + 1 - i] * rightvalue

                leftpoint = CartesianIndex(setindex(point.I, total - O + i, axis))
                leftvalue = evaluate(leftpoint, leftblock, field, rest...)
                valuerhs -= valuestencilright[i] * leftvalue
                derivativerhs -= derivativestencilright[i] * leftvalue
            end

            # For each known value in the on the other domain 
            for i in 1:(num - 1)
                leftvalue = unknownleft[i]
                valuerhs += valuestencilleft[L + 1- i] * leftvalue
                derivativerhs += derivativestencilleft[1 + i] * leftvalue

                rightvalue = unknownright[i]
                valuerhs -= valuestencilright[O + i] * rightvalue
                derivativerhs -= derivativestencilright[O + i] * rightvalue
            end

            valuematrix = (-valuestencilleft[begin], valuestencilright[end])
            derivativematrix = (-derivativestencilleft[begin], derivativestencilright[end])

            matrix = SA[valuematrix[1] valuematrix[2]; derivativematrix[1] derivativematrix[2]]
            rhs = SA[valuerhs, derivativerhs]

            result = matrix \ rhs

            unknownleft = setindex(unknownleft, result[1], num)
            unknownright = setindex(unknownright, result[2], num)
        end

        unknowns = side ? unknownright : unknownleft

        return sum(coefficients.values .* unknowns)
    end
end
