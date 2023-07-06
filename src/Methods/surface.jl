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

function Operators.interface(point::CartesianIndex{N}, block::TreeBlock{N, T}, field::TreeField{N, T}, face::FaceIndex{N}, coefs::InterfaceCoefs{T, L, O}, operator::Operator{T}, rest::Operator{T}...) where {N, T, L, O}
    surface = block.surface
    tree = surface.tree
    node = block.node

    axis = faceaxis(face)
    side = faceside(face)
    total = nodecells(tree)[axis]

    neighbor = tree.neighbors[node][face]

    if neighbor < 0
        # Handle boundary conditions
        unknown = ntuple(i -> zero(T), Val(L))

        for num in 1:L
            valuecoefs = boundary_value_coefs(operator, num, side)

            valuerhs = zero(T)

            for i in 1:O
                if side
                    offpoint = CartesianIndex(setindex(point, cell_to_point(i), axis))
                else
                    offpoint = CartesianIndex(setindex(point, cell_to_point(total + 1 - i), axis))
                end

                value = evaluate_point(offpoint, block, field, rest...)
                valuerhs -= valuecoefs.interior[i] * value
            end

            for i in 1:(num - 1)
                value = unknown[i]
                valuerhs -= valuecoefs.exterior[i] * value
            end

            unknown = setindex(unknown, valuerhs / value.exterior[end], num) 
        end

        return sum(coefs.values .* unknowns)
    end

    # Get leftblock and rightblock
    @assert neighbor > 0 && surface.nodes[neighbor] > 0

    # Block to the left of the boundary
    leftblock = side ? block : TreeBlock(surface, neighbor)
    # Block to the right of the boundary
    rightblock = side ? TreeBlock(surface, neighbor) : block

    # Also set leftpoint and rightpoint
    leftpoint = point.I
    rightpoint = point.I

    # Everything else should be independent
    unknownleft = ntuple(i -> zero(T), Val(L))
    unknownright = ntuple(i -> zero(T), Val(L))

    for num in 1:L
        valuerhs = zero(T)
        derivativerhs = zero(T)

        valuecoefsleft = boundary_value_coefs(operator, num, false)
        valuecoefsright = boundary_value_coefs(operator, num, true)
        derivativecoefsleft = boundary_derivative_coefs(operator, num, false)
        derivativecoefsright = boundary_derivative_coefs(operator, num, true)

        # For each value on their own domain
        for i in 1:O
           rightvalue = evaluate_point(CartesianIndex(setindex(rightpoint, cell_to_point(i), axis)), rightblock, field, rest...)
           valuerhs += valuecoefsleft.interior[i] * rightvalue
           derivativerhs += derivativecoefsleft.interior[i] * rightvalue

           leftvalue = evaluate(CartesianIndex(setindex(leftpoint, cell_to_point(total + 1 - i), axis)), leftblock, field, rest...)
           valuerhs -= valuecoefsright.interior[i] * leftvalue
           derivativerhs -= derivativecoefsright.interior[i] * leftvalue
        end

        # For each known value in the on the other domain 
        for i in 1:(num - 1)
            leftvalue = unknownleft[i]
            valuerhs += valuecoefsleft.exterior[i] * leftvalue
            derivativerhs += derivativecoefsleft.exterior[i] * leftvalue

            rightvalue = unknownright[i]
            valuerhs -= valuecoefsright.exterior[i] * rightvalue
            derivativerhs -= derivativecoefsright.exterior[i] * rightvalue
        end

        matrix = SA[-valuecoefsleft.exterior[end] valuecoefsright.exterior[end]; 
                    -derivativecoefsleft.exterior[end]  derivativecoefsright.exterior[end]]
        rhs = SA[valuerhs, derivativerhs]
        result = matrix \ rhs

        unknownleft = setindex(unknownleft, result[1], num)
        unknownright = setindex(unknownright, result[2], num)
    end

    unknowns = side ? unknownright : unknownleft

    return sum(coefs.values .* unknowns)
end
