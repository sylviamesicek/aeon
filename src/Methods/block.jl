
struct TreeBlock{N, T} <: Block{N, T}
    surface::TreeSurface{N, T}
    node::Int
end

function Operators.blockcells(block::TreeBlock{N, T}) where {N, T}
    refinement::Int = block.surface.tree.refinement
    ntuple(i -> 2^refinement, Val(N))
end

Operators.blockbounds(block::TreeBlock) = block.surface.tree.bounds[block.node]
 
struct TreeField{N, T, S <: AbstractVector{T}} <: Field{N, T}
    values::S
end

TreeField{N}(values::AbstractVector{T}) where {N, T} = TreeField{N, T, typeof(values)}(values)

Base.similar(surface::TreeSurface{N, T}) where {N, T} = TreeField{N}(Vector{T}(undef, surface.total))

function Operators.value(field::TreeField{N, T}, block::TreeBlock{N, T}, cell::CartesianIndex{N, T}) where {N, T}
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

function Operators.setvalue!(field::TreeField{N, T}, value::T, block::TreeBlock{N, T}, cell::CartesianIndex{N, T}) where {N, T}
    # Find offset 
    offset = block.surface.offsets[block.node]
    # Build map to linear
    linear = LinearIndices(nodecells(block.surface.tree))
    # Get global ptr
    ptr = linear[cell] + offset
    # Access value
    field.values[ptr] = value
end

# function Operators.interface(::Val{S}, point::CartesianIndex{N}, block::TreeBlock{N, T}, field::TreeField{N, T}, coefs::NTuple{O, T}, extent::Int, opers::NTuple{L, Operator{T}}) where {N, T, O, S, L}
#     face = FaceIndex{N}(L, S)
#     index = point[L]
#     cell_total = blockcells(block)[L]

#     surface = block.surface
#     tree = surface.tree
#     node = block.node

#     neighbor = tree.neighbors[node][face]

#     valuestencils = interface_value_stencils(opers[L], Val(S))
#     derivativestencils = interface_derivative_stencils(opers[L], Val(S))

#     if neighbor < 0
#         # Handle boundary conditions
#         boundary_unknowns::NTuple{O, T} = ntuple(i -> zero(T), Val(O))
#         boundary_result = zero(T)

#         if face == FaceIndex{N}(1, false) || face == FaceIndex{N}(2, false)
#             for i in 1:(O - extent)
#                 stencil = derivativestencils[i]
#                 rhs = zero(T)
#                 rhs -= evaluate_interface_interior(Val(S), point, block, field, stencil, opers)
#                 rhs -= evaluate_interface_exterior(stencil, boundary_unknowns, i - 1)
    
#                 unknown = rhs / stencil.edge
#                 boundary_unknowns = setindex(boundary_unknowns, unknown, i) 
#                 boundary_result += coefs[extent + i] * unknown
#             end
#         else
#             for i in 1:(O - extent)
#                 stencil = valuestencils[i]
#                 rhs = zero(T)
#                 rhs -= evaluate_interface_interior(Val(S), point, block, field, stencil, opers)
#                 rhs -= evaluate_interface_exterior(stencil, boundary_unknowns, i - 1)
    
#                 unknown = rhs / stencil.edge
#                 boundary_unknowns = setindex(boundary_unknowns, unknown, i) 
#                 boundary_result += coefs[extent + i] * unknown
#             end
#         end
        
        

#         return boundary_result
#     end

#     # Get leftblock and rightblock
#     # @assert neighbor > 0 && surface.nodes[neighbor] > 0

#     # if neighbor > 0 && tree.children[neighbor] > 0
#     #     # We are coarse and the neighbor is more refined. 
#     #     cell_index = point_to_cell(index)
#     # end

#     otherS = !S
#     otherblock = TreeBlock(surface, neighbor)
#     othervaluestencils = interface_value_stencils(opers[L], Val(otherS))
#     otherderivativestencils = interface_derivative_stencils(opers[L], Val(otherS))

#     # Everything else should be independent
#     unknowns::NTuple{O, T} = ntuple(i -> zero(T), Val(O))
#     otherunknowns::NTuple{O, T} = ntuple(i -> zero(T), Val(O))

#     interface_result::T = zero(T)

#     for i in 1:(O - extent)
#         valuestencil = valuestencils[i]
#         derivativestencil = derivativestencils[i]
#         othervaluestencil = othervaluestencils[i]
#         otherderivativestencil = otherderivativestencils[i]

#         valuerhs = evaluate_interface_interior(Val(otherS), point, otherblock, field, othervaluestencil, opers)
#         valuerhs += evaluate_interface_exterior(othervaluestencil, otherunknowns, i - 1)
#         valuerhs -= evaluate_interface_interior(Val(S), point, block, field, valuestencil, opers)
#         valuerhs -= evaluate_interface_exterior(valuestencil, unknowns, i - 1)

#         derivativerhs = evaluate_interface_interior(Val(otherS), point, otherblock, field, otherderivativestencil, opers)
#         derivativerhs += evaluate_interface_exterior(otherderivativestencil, otherunknowns, i - 1)
#         derivativerhs -= evaluate_interface_interior(Val(S), point, block, field, derivativestencil, opers)
#         derivativerhs -= evaluate_interface_exterior(derivativestencil, unknowns, i - 1)

#         matrix = SA{T}[
#             valuestencil.edge  (-othervaluestencil.edge); 
#             derivativestencil.edge  (-otherderivativestencil.edge)
#         ]

#         rhs::SVector{N, T} = SA[valuerhs, derivativerhs]
#         unknown_result::SVector{N, T} = matrix \ rhs

#         unknowns = setindex(unknowns, unknown_result[1], i)
#         otherunknowns = setindex(otherunknowns, unknown_result[2], i)
        
#         interface_result += coefs[extent + i] * unknown_result[1]
#     end

#     return interface_result
# end
