#####################
## Block ############
#####################

export TreeBlock

struct TreeBlock{N, T} <: Block{N, T}
    surface::TreeSurface{N, T}
    node::Int
end

function Operators.blockcells(block::TreeBlock{N, T}) where {N, T}
    refinement::Int = block.surface.tree.refinement
    ntuple(i -> 2^refinement, Val(N))
end

Operators.blockbounds(block::TreeBlock) = block.surface.tree.bounds[block.node]

######################
## Boundary ##########
######################

export BoundaryKind, BC, BoundaryCondition, Diritchlet, Nuemann, Flatness

@enum BoundaryKind begin
    Diritchlet = 1
    Nuemann = 2
    Flatness = 3
end

struct BoundaryCondition{T} 
    kind::BoundaryKind
    homogenous::T

    BoundaryCondition(kind::BoundaryKind, homogenous::T) where T = new{T}(kind, homogenous)
end

BC(kind::BoundaryKind, homogenous) = BoundaryCondition(kind, homogenous)

######################
## Tree Field ########
######################

export TreeField
 
struct TreeField{N, T, V <: AbstractVector{T}, F} <: Field{N, T}
    values::V
    boundaries::HyperFaces{N, BoundaryCondition{T}, F}

    TreeField(values::AbstractVector{T}, boundaries::HyperFaces{N, BoundaryCondition{T}}) where {N, T} = new{N, T, typeof(values), 2N}(values, boundaries)
end

TreeField(::UndefInitializer, surface::TreeSurface{N, T}, boundaries::HyperFaces{N, BoundaryCondition{T}}) where {N, T} = TreeField(Vector{T}(undef, surface.total), boundaries)

# Necessary for Operators implementation as field
function Operators.value(field::TreeField{N, T}, block::TreeBlock{N, T}, cell::CartesianIndex{N}) where {N, T}
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

function Operators.setvalue!(field::TreeField{N, T}, value::T, block::TreeBlock{N, T}, cell::CartesianIndex{N}) where {N, T}
    # Find offset 
    offset = block.surface.offsets[block.node]
    # Build map to linear
    linear = LinearIndices(blockcells(block))
    # Get global ptr
    ptr = linear[cell] + offset
    # Access value
    field.values[ptr] = value
end

# Necessary for implementation as AbstractArray

Base.size(field::TreeField) = size(field.values)
Base.length(field::TreeField) = length(field.values)
Base.getindex(field::TreeField, i::Int) = field.values[i]
Base.setindex!(field::TreeField, v, i::Int) = setindex!(field.values, v, i)
Base.eachindex(field::TreeField) = eachindex(field.values)
Base.IndexStyle(field::TreeField) = IndexStyle(field.values)


#########################
## Evaluation ###########
#########################

function Operators.interface_value(field::TreeField{N, T}, block::TreeBlock{N, T}, cell::CartesianIndex{N}, ::AbstractBasis{T}, ::Val{O}, ::Val{I}) where {N, T, O, I}
    value = zero(T)

    for axis in 1:N
        if I[axis] == 0
            continue
        end

        boundary = field.boundaries[FaceIndex{N}(axis, I[axis] > 0)]

        if boundary.kind == Diritchlet
            value += one(T)
        elseif boundary.kind == Flatness
            center = cellcenter(block, cell)
            trans = blocktransform(block)
            r = norm(trans(center))
            value += one(T) / r
        end
    end

    value
end

function Operators.interface_gradient(field::TreeField{N, T}, block::TreeBlock{N, T}, ::CartesianIndex{N}, ::AbstractBasis{T}, ::Val{O}, ::Val{I}) where {N, T, O, I}
    SVector(ntuple(Val(N)) do axis
        if I[axis] == 0
            return zero(T)
        end

        value = zero(T)

        boundary = field.boundaries[FaceIndex{N}(axis, I[axis] > 0)]

        h⁻¹ = blockcells(block)[axis] / blockbounds(block).widths[axis]

        if boundary.kind == Nuemann || boundary.kind == Flatness
            value += one(T) * h⁻¹
        end
        
        value
    end)
end

function Operators.interface_homogenous(field::TreeField{N, T}, block::TreeBlock{N, T}, cell::CartesianIndex{N}, ::AbstractBasis{T}, ::Val{O}, ::Val{I}) where {N, T, O, I}
    homogenous = zero(T)

    for axis in 1:N
        if I[axis] == 0
            continue
        end

        boundary = field.boundaries[FaceIndex{N}(axis, I[axis] > 0)]

        if boundary.kind == Diritchlet || boundary.kind == Nuemann
            homogenous += boundary.homogenous
        elseif boundary.kind == Flatness
            center = cellcenter(block, cell)
            trans = blocktransform(block)
            r = norm(trans(center))
            homogenous += boundary.homogenous / r
        end
    end

    homogenous
end



# function Operators.interface_condition(field::TreeField{N, T}, block::TreeBlock{N, T}, cell::CartesianIndex{N}, basis::AbstractBasis{T}, ::Val{O}, ::Val{I}) where {N, T, O, I}
#     _interface_product(field, block, cell, basis, Val(O), map(Val, I)...)
# end

# function _interface_product(field::Field{N, T}, block::Block{N, T}, cell::CartesianIndex{N}, basis::AbstractBasis{T}, ::Val{O}) where {N, T, O}
#     blockprolong(field, block, map(CellIndex, cell.I), basis, Val(O))
# end

# function _interface_product(field::Field{N, T}, block::Block{N, T}, cell::CartesianIndex{N}, basis::AbstractBasis{T}, ::Val{O}, ::Val{0}, rest::Vararg{Val, L}) where {N, T, O, L}  
#     _interface_product(field, block, cell, basis, Val(O), rest...)
# end

# function _interface_product(field::Field{N, T}, block::Block{N, T}, cell::CartesianIndex{N}, basis::AbstractBasis{T}, ::Val{O}, ::Val{E}, rest::Vararg{Val, L}) where {N, T, O, L, E}    
#     condition = field.boundaries[FaceIndex{N}(N - L, E > 0)]

#     if condition.kind == Diritchlet
#         _interface_diritchlet(condition.coef, field, block, cell, basis, Val(O), Val(E), rest...)
#     elseif condition.kind == Nuemann
#         _interface_nuemann(condition.coef, field, block, cell, basis, Val(O), Val(E), rest...)
#     else
#         _interface_flatness(condition.coef, field, block, cell, basis, Val(O), Val(E), rest...)
#     end
# end

# @generated function _interface_diritchlet(coef::T, field::Field{N, T}, block::Block{N, T}, cell::CartesianIndex{N}, basis::AbstractBasis{T}, ::Val{O}, ::Val{E}, rest::Vararg{Val, L}) where {N, T, O, E, L}
#     quote
#         interior = compute_interior(field, block, cell, basis, Val($O), Val($E), rest...)
#         exterior = Base.@ntuple $O i -> zero($T)

#         Base.@nexprs $O i -> begin
#             stencil_i = interface_value_stencil(basis, Val($(2O)), Val(i), Val($(E > 0)))
#             rhs_i = coef - interface_apply_stencil(interior, exterior, stencil_i, Val($(E > 0)))
#             edge_i = interface_edge_stencil(stencil_i, Val($(E > 0)))

#             exterior = setindex(exterior, rhs_i/edge_i, i)
#         end

#         stencil = interface_value_stencil(basis, Val($(2O)), Val($O), Val($(E > 0)))
#         interface_apply_stencil(interior, exterior, stencil, Val($(E > 0)))
#     end
# end

# @generated function _interface_nuemann(coef::T, field::Field{N, T}, block::Block{N, T}, cell::CartesianIndex{N}, basis::AbstractBasis{T}, ::Val{O}, ::Val{E}, rest::Vararg{Val, L}) where {N, T, O, E, L}
#     axis = N - L
#     quote
#         interior = compute_interior(field, block, cell, basis, Val($O), Val($E), rest...)
#         exterior = Base.@ntuple $O i -> zero($T)

#         h⁻¹ = blockcells(block)[$axis] / blockbounds(block).widths[$axis]

#         Base.@nexprs $O i -> begin
#             stencil_i = interface_derivative_stencil(basis, Val($(2O)), Val(i), Val($(E > 0)))
#             rhs_i = coef - interface_apply_stencil(interior, exterior, stencil_i, Val($(E > 0))) * h⁻¹
#             edge_i = interface_edge_stencil(stencil_i, Val($(E > 0))) * h⁻¹

#             exterior = setindex(exterior, rhs_i/edge_i, i)
#         end
        
#         stencil = interface_value_stencil(basis, Val($(2O)), Val($O), Val($(E > 0)))
#         interface_apply_stencil(interior, exterior, stencil, Val($(E > 0)))
#     end
# end

# @generated function _interface_flatness(coef::T, field::Field{N, T}, block::Block{N, T}, cell::CartesianIndex{N}, basis::AbstractBasis{T}, ::Val{O}, ::Val{E}, rest::Vararg{Val, L}) where {N, T, O, E, L}
#     axis = N - L
#     quote
#         interior = compute_interior(field, block, cell, basis, Val($O), Val($E), rest...)
#         exterior = Base.@ntuple $O i -> zero($T)

#         trans = blocktransform(block)
#         center = cellcenter(block, cell)
#         r = norm(trans(center)) # Compute distance

#         h⁻¹ = blockcells(block)[$axis] / blockbounds(block).widths[$axis]

#         Base.@nexprs $O i -> begin
#             valuestencil_i = interface_value_stencil(basis, Val($(2O)), Val(i), Val($(E > 0)))
#             valuerhs_i = interface_apply_stencil(interior, exterior, valuestencil_i, Val($(E > 0))) * h⁻¹
#             valueedge_i = interface_edge_stencil(valuestencil_i, Val($(E > 0))) * h⁻¹

#             derivstencil_i = interface_derivative_stencil(basis, Val($(2O)), Val(i), Val($(E > 0)))
#             derivrhs_i = interface_apply_stencil(interior, exterior, derivstencil_i, Val($(E > 0))) * h⁻¹
#             derivedge_i = interface_edge_stencil(derivstencil_i, Val($(E > 0))) * h⁻¹

#             rhs_i = (coef + valuerhs_i) / r - derivrhs_i
#             edge_i = derivedge_i - valueedge_i / r 

#             exterior = setindex(exterior, rhs_i/edge_i, i)
#         end

#         stencil = interface_value_stencil(basis, Val($(2O)), Val($O), Val($(E > 0)))
#         interface_apply_stencil(interior, exterior, stencil, Val($(E > 0)))
#     end
# end

# #############################
# ## Helpers ##################
# #############################

# function interface_value_stencil(basis::AbstractBasis{T}, ::Val{I}, ::Val{E}, ::Val{S}) where {T, E, S, I}
#     if S 
#         vertex_value_stencil(basis, Val(I), Val(E))
#     else
#         vertex_value_stencil(basis, Val(E), Val(I))
#     end
# end

# function interface_derivative_stencil(basis::AbstractBasis{T}, ::Val{I}, ::Val{E}, ::Val{S}) where {T, E, S, I}
#     if S 
#         vertex_derivative_stencil(basis, Val(I), Val(E))
#     else
#         vertex_derivative_stencil(basis, Val(E), Val(I))
#     end
# end

# function interface_apply_stencil(interior::NTuple{I, T}, exterior::NTuple{E, T}, stencil::VertexStencil{T}, ::Val{S}) where {T, I, E, S}
#     result = zero(T)

#     if S
#         for j in eachindex(stencil.left)
#             result += stencil.left[j] * interior[j]
#         end

#         for j in eachindex(stencil.right)
#             result += stencil.right[j] * exterior[j]
#         end
#     else
#         for j in eachindex(stencil.right)
#             result += stencil.right[j] * interior[j]
#         end

#         for j in eachindex(stencil.left)
#             result += stencil.left[j] * exterior[j]
#         end
#     end

#     return result
# end

# function interface_edge_stencil(stencil::VertexStencil{T}, ::Val{S}) where {T, S}
#     if S
#         return stencil.right[end]
#     else
#         return stencil.left[end]
#     end
# end

# function compute_interior(field::Field{N, T}, block::Block{N, T}, cell::CartesianIndex{N}, basis::AbstractBasis{T}, ::Val{O}, ::Val{E}, rest::Vararg{Val, L}) where {N, T, O, E, L}
#     axis = N - L

#     # Fill interior
#     return ntuple(Val(2O)) do i
#         offcell = CartesianIndex(setindex(cell.I, cell[axis] - E * i, axis))
#         _interface_product(field, block, offcell, basis, Val(O), rest...)
#     end
# end


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
