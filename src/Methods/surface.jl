export TreeSurface, TreeField, TreeBlock
export setfieldvalue!

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

struct TreeField{N, T} <: Field{N, T}
    values::Vector{T}
end

Base.similar(surface::TreeSurface{N, T}) where {N, T} = TreeField{N, T}(Vector{T}(undef, surface.total))

struct TreeBlock{N, T} <: Block{N, T}
    surface::TreeSurface{N, T}
    node::Int
end

function setfieldvalue!(cell::CartesianIndex{N}, block::TreeBlock{N, T}, field::TreeField{N, T}, value::T) where {N, T}
    # Find offset 
    offset = block.surface.offsets[block.node]
    # Build map to linear
    linear = LinearIndices(nodecells(block.surface.tree))
    # Get global ptr
    ptr = linear[cell] + offset
    # Access value
    field.values[ptr] = value
end

Operators.blockcells(block::TreeBlock) = nodecells(block.surface.tree)
Operators.blockbounds(block::TreeBlock) = block.surface.tree.bounds[block.node]
 
function Operators.evaluate(cell::CartesianIndex{N}, block::TreeBlock{N, T}, field::TreeField{N, T}) where {N, T}
    # Find offset 
    offset = block.surface.offsets[block.node]
    # Build map to linear
    linear = LinearIndices(blockcells(block))
    # Get global ptr
    ptr = linear[cell] + offset
    # Access value
    return field.values[ptr]
end

function Operators.interface(point::CartesianIndex{N}, block::TreeBlock{N, T}, field::TreeField{N, T}, inter::Interface{T, F, E}, operator::Operator{T}, rest::Operator{T}...) where {N, T, F, E}
    surface = block.surface
    tree = surface.tree
    node = block.node

    side = faceside(F)

    neighbor = tree.neighbors[node][F]

    if neighbor < 0
        # Handle boundary conditions
        unknowns = ntuple(i -> zero(T), Val(E))

        for extent in 1:E
            stencil = interface_value_stencil(operator, extent, side)
            valuerhs = -evaluate_interface_interior(point, block, field, F, stencil, rest...)

            for i in eachindex(stencil.exterior)
                valuerhs -= stencil.exterior[i] * unknowns[i]
            end

            unknowns = setindex(unknowns, valuerhs / stencil.edge, extent) 
        end

        return sum(inter.coefficients .* unknowns)
    end

    # Get leftblock and rightblock
    @assert neighbor > 0 && surface.nodes[neighbor] > 0

    otherblock = TreeBlock(surface, neighbor)
    otherF = facereverse(F)

    # Everything else should be independent
    unknowns = ntuple(i -> zero(T), Val(E))
    otherunknowns = ntuple(i -> zero(T), Val(E))

    for extent in 1:E
        valuestencil = interface_value_stencil(operator, extent, side)
        derivativestencil = interface_derivative_stencil(operator, extent, side)

        othervaluestencil = interface_value_stencil(operator, extent, !side)
        otherderivativestencil = interface_derivative_stencil(operator, extent, !side)

        valuerhs = -evaluate_interface_interior(point, block, field, F, valuestencil, rest...) + evaluate_interface_interior(point, otherblock, field, otherF, othervaluestencil, rest...)
        derivativerhs = -evaluate_interface_interior(point, block, field, F, derivativestencil, rest...) + evaluate_interface_interior(point, otherblock, field, otherF, otherderivativestencil, rest...)

        for i in eachindex(valuestencil.exterior)
            valuerhs -= valuestencil.exterior[i] * unknowns[i]
        end

        for i in eachindex(derivativestencil.exterior)
            derivativerhs -= derivativestencil.exterior[i] * unknowns[i]
        end

        for i in eachindex(othervaluestencil.exterior)
            valuerhs += othervaluestencil.exterior[i] * otherunknowns[i]
        end

        for i in eachindex(otherderivativestencil.exterior)
            derivativerhs += otherderivativestencil.exterior[i] * otherunknowns[i]
        end

        matrix = SA[
            valuestencil.edge  (-othervaluestencil.edge); 
            derivativestencil.edge  (-otherderivativestencil.edge)
        ]

        rhs = SA[valuerhs, derivativerhs]
        result = matrix \ rhs

        unknowns = setindex(unknowns, result[1], extent)
        otherunknowns = setindex(otherunknowns, result[2], extent)
    end

    return sum(inter.coefficients .* unknowns)
end
