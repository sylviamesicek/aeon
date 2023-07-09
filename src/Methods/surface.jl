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

struct TreeField{N, T, S <: AbstractVector{T}} <: Field{N, T}
    values::S
end

TreeField{N}(values::AbstractVector{T}) where {N, T} = TreeField{N, T, typeof(values)}(values)

Base.similar(surface::TreeSurface{N, T}) where {N, T} = TreeField{N}(Vector{T}(undef, surface.total))

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

function Operators.blockcells(block::TreeBlock{N, T}) where {N, T}
    refinement::Int = block.surface.tree.refinement
    ntuple(i -> 2^refinement, Val(N))
end

Operators.blockbounds(block::TreeBlock) = block.surface.tree.bounds[block.node]
 
function Operators.evaluate(cell::CartesianIndex{N}, block::TreeBlock{N, T}, field::TreeField{N, T}) where {N, T}
    # Find offset 
    offset = block.surface.offsets[block.node]
    cells = blockcells(block)
    # Build map to linear
    linear = LinearIndices(cells)
    # Get global ptr
    ptr = linear[cell] + offset
    # Access value
    return field.values[ptr]
end

function Operators.interface(::Val{S}, point::CartesianIndex{N}, block::TreeBlock{N, T}, field::TreeField{N, T}, coefs::NTuple{O, T}, extent::Int, opers::NTuple{L, Operator{T}}) where {N, T, O, S, L}
    face = FaceIndex{N}(L, S)
    remaining = ntuple(i -> opers[i], Val(L - 1))

    surface = block.surface
    tree = surface.tree
    node = block.node

    neighbor = tree.neighbors[node][face]

    valuestencils = interface_value_stencils(opers[L], Val(S))
    derivativestencils = interface_derivative_stencils(opers[L], Val(S))

    if neighbor < 0
        # Handle boundary conditions
        unknowns = ntuple(i -> zero(T), Val(O))
        result = zero(T)

        for i in 1:(O - extent)
            valuestencil = valuestencils[i]
            valuerhs = -evaluate_interface_interior(Val(S), point, block, field, valuestencil, remaining)
            valuerhs -= evaluate_interface_exterior(valuestencil, unknowns, i - 1)

            unknown = valuerhs / valuestencil.edge
            unknowns = setindex(unknowns, unknown, i) 
            result += coefs[extent + i] * unknown
        end

        return result
    end

    # Get leftblock and rightblock
    # @assert neighbor > 0 && surface.nodes[neighbor] > 0

    otherS = !S
    otherblock = TreeBlock(surface, neighbor)
    othervaluestencils = interface_value_stencils(opers[L], Val(otherS))
    otherderivativestencils = interface_derivative_stencils(opers[L], Val(otherS))

    # Everything else should be independent
    unknowns = ntuple(i -> zero(T), Val(O))
    otherunknowns = ntuple(i -> zero(T), Val(O))

    res = zero(T)

    for i in 1:(O - extent)
        valuestencil = valuestencils[i]
        derivativestencil = derivativestencils[i]
        othervaluestencil = othervaluestencils[i]
        otherderivativestencil = otherderivativestencils[i]

        valuerhs = evaluate_interface_interior(Val(otherS), point, otherblock, field, othervaluestencil, rest...)
        valuerhs += evaluate_interface_exterior(othervaluestencil, otherunknowns, i - 1)
        valuerhs -= evaluate_interface_interior(Val(S), point, block, field, valuestencil, rest...)
        valuerhs -= evaluate_interface_exterior(valuestencil, unknowns, i - 1)

        derivativerhs = evaluate_interface_interior(Val(otherS), point, otherblock, field, otherderivativestencil, rest...)
        derivativerhs += evaluate_interface_exterior(otherderivativestencil, otherunknowns, i - 1)
        derivativerhs -= evaluate_interface_interior(Val(S), point, block, field, derivativestencil, rest...)
        derivativerhs -= evaluate_interface_exterior(derivativestencil, unknowns, i - 1)

        matrix = SA[
            valuestencil.edge  (-othervaluestencil.edge); 
            derivativestencil.edge  (-otherderivativestencil.edge)
        ]

        rhs = SA[valuerhs, derivativerhs]
        result = matrix \ rhs

        unknowns = setindex(unknowns, result[1], i)
        otherunknowns = setindex(otherunknowns, result[2], i)

        res += coefs[extent + i] * result[1]
    end

    return res
end
