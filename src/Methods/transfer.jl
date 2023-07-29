############################
## Transfer ################
############################

export transfer_to_block!

"""
Transfers data from vector to block a block, with the given order of accuracy.
"""
function transfer_to_block!(f::F, block::ArrayBlock{N, T, O}, values::AbstractVector{T}, basis::AbstractBasis{T}, mesh::Mesh{N, T}, dofs::DoFManager{N, T}, level::Int, node::Int) where {N, T, O, F <: Function}
    # Fill interior 
    offset = nodeoffset(dofs, level, node)
    
    fill_interior_from_linear!(block) do i
        values[offset + i]
    end

    # Fill boundaries
    fill_boundaries!(block, basis) do cell, i
        @inline compute_block_boundary(TransferBlock{O}(f, values, basis, mesh, dofs, level, node, i), cell)
    end
end

struct TransferBlock{N, T, O, I, F, V, B} <: AbstractBlock{N, T, O} 
    values::V
    f::F
    basis::B
    mesh::Mesh{N, T}
    dofs::DoFManager{N, T}
    level::Int
    node::Int

    TransferBlock{O}(f::Function, values::AbstractVector{T}, basis::AbstractBasis{T}, mesh::Mesh{N, T}, dofs::DoFManager{N, T}, level::Int, node::Int, ::Val{I}) where {N, T, O, I} = new{N, T, O, I, typeof(f), typeof(values), typeof(basis)}(values, f, basis, mesh, dofs, level, node)
end

blocktotal(block::TransferBlock) = nodecells(block.mesh, block.level)
blockcells(block::TransferBlock) = nodecells(block.mesh, block.level)
blockvalue(block::TransferBlock{N}, cell::CartesianIndex{N}) where N = compute_block_boundary(block, cell)

"""
A god awful monstrosity of a function which computes target values for each exterior cell.
"""
@generated function compute_block_boundary(block::TransferBlock{N, T, O, I}, cell::CartesianIndex{N}) where {N, T, O, I}
    size = ntuple(i -> ifelse(I[i] ≠ 0, O, 1), N)
    len = prod(size)

    reversed::Tuple = map(Val, reverse(I))

    quote
        boundary = _compute_block_boundary(block, cell.I, $(reversed...))
        SArray{Tuple{$(size...)}, $T, $N, $len}(boundary)
    end
end

function _compute_block_boundary(block::TransferBlock{N, T}, cell::NTuple{N, Int}) where {N, T}
    # The base case in the recursion
    cells = nodecells(block.mesh, block.level)
    offset = nodeoffset(block.dofs, block.level, block.node)
    linear = LinearIndices(cells)
    return block.values[offset + linear[cell...]]
end

function _compute_block_boundary(block::TransferBlock{N, T}, cell::NTuple{N, Int}, ::Val{0}, rest::Vararg{Val, M}) where {N, T, M}
    # This axis does not need to be extended
    _compute_block_boundary(block, cell, rest...)
end

@generated function _compute_block_boundary(block::TransferBlock{N, T, O, G}, cell::NTuple{N, Int}, ::Val{I}, rest::Vararg{Val, M}) where {N, T, O, I, G, M}
    axis = M + 1
    side = I > 0
    face = FaceIndex{N}(axis, side)

    quote
        cells = nodecells(block.mesh, block.level)
        neighbor = nodeneighbors(block.mesh, block.level, block.node)[$face]
        transform = nodetransform(block.mesh, block.level, block.node)
        position = cellposition(block.mesh, block.level, CartesianIndex(cell))
        offset = nodeoffset(block.dofs, block.level, block.node)

        if neighbor < 0
            gpos = transform(position .+ G .* 1 ./ (2 .* cells))
            condition = block.f(gpos, $face)

            return tuple_flatten(_physical_boundary(condition, block, cell, Val(I), rest...))
        elseif neighbor == 0
            # Neighbor is coarser
            error("Coarse neighbor unimplemented")
        elseif nodechildren(block.mesh, block.level, neighbor) == -1
            # Neighbor is same level
            return tuple_flatten(_identity_interface(block, cell, Val(I), rest...))
        else
            # Neighbor is more refined
            error("Refined  neighbor unimplemented")
        end
    end
end

@generated function _physical_boundary(condition::BoundaryCondition{T}, block::TransferBlock{N, T, O}, cell::NTuple{N, Int}, ::Val{I}, rest::Vararg{Val, M}) where {N, T, O, I, M}
    axis = M + 1
    side = I > 0
   
    quote
        total = nodecells(block.mesh, block.level)[$axis]
        width = nodebounds(block.mesh, block.level, block.node).widths[$axis]

        ccell = setindex(cell, $side ? total : 1, $axis)
        center = _compute_block_boundary(block, ccell, rest...)

        interior = Base.@ntuple $(2O) j -> begin
            icell_j = setindex(cell, $side ? total - j : 1 + j, $axis)
            _compute_block_boundary(block, icell_j, rest...)
        end

        Base.@nexprs $O i -> begin
            # Value
            vstencil_i = Stencil(block.basis, $side ? VertexValue{$(2O + 1), i, false}() : VertexValue{i, $(2O + 1), true}())
            vresult_i = vstencil_i.center .* center

            interior_vstencil_i = $side ? vstencil_i.left : vstencil_i.right
            exterior_vstencil_i = $side ? vstencil_i.right : vstencil_i.left

            Base.@nexprs $(2O) j -> vresult_i = vresult_i .+ interior_vstencil_i[j] .* interior[j]
            Base.@nexprs (i - 1) j -> vresult_i = vresult_i .+ exterior_vstencil_i[j] .* unknown_j

            # Derivative
            dstencil_i = Stencil(block.basis, $side ? VertexDerivative{$(2O + 1), i, false}() : VertexDerivative{i, $(2O + 1), true}())
            dresult_i = dstencil_i.center .* center

            interior_dstencil_i = $side ? dstencil_i.left : dstencil_i.right
            exterior_dstencil_i = $side ? dstencil_i.right : dstencil_i.left

            Base.@nexprs $(2O) j -> dresult_i = dresult_i .+ interior_dstencil_i[j] .* interior[j]
            Base.@nexprs (i - 1) j -> dresult_i = dresult_i .+ exterior_dstencil_i[j] .* unknown_j

            # Transform normal derivative
            normal_to_global = I * total / width

            vnumerator = condition.value .* vresult_i
            dnumerator = condition.normal .* normal_to_global .* dresult_i
            rnumerator = condition.rhs

            vdenominator = condition.value * exterior_vstencil_i[end]
            ddenominator = condition.normal * normal_to_global * exterior_dstencil_i[end] 

            unknown_i = (rnumerator .- vnumerator .- dnumerator) ./ (vdenominator + ddenominator)

            vapplied_i = vresult_i .+ exterior_vstencil_i[end] .* unknown_i
        end

        return Base.@ntuple $O i -> vapplied_i
    end
end

@generated function _identity_interface(block::TransferBlock{N, T, O}, cell::NTuple{N, Int}, ::Val{I}, rest::Vararg{Val, M}) where {N, T, O, I, M}
    axis = M + 1
    side = I > 0
    face = FaceIndex{N}(axis, side)
    
    quote
        neighbor = nodeneighbors(block.mesh, block.level, block.node)[$face]
        total = nodecells(block.mesh, block.level)[$axis]

        ccell = setindex(cell, $side ? total : 1, $axis)
        center = _compute_block_boundary(block, ccell, rest...)

        interior = Base.@ntuple $(2O) j -> begin
            icell_j = setindex(cell, $side ? total - j : 1 + j, $axis)
            _compute_block_boundary(block, icell_j, rest...)
        end

        exterior = Base.@ntuple $(O) j -> begin
            ecell_j = setindex(cell, $side ? j : total + 1 - j, $axis)
            _compute_block_boundary(TransferBlock{O}(block.f, block.values, block.basis, block.mesh, block.dofs, block.level, neighbor, Val(I)), ecell_j, rest...)
        end

        return Base.@ntuple $O i -> begin
            stencil = Stencil(block.basis, $side ? VertexValue{$(2O + 1), i, false}() : VertexValue{i, $(2O + 1), true}())

            interior_stencil = $side ? stencil.left : stencil.right
            exterior_stencil = $side ? stencil.right : stencil.left

            result = stencil.center .* center

            Base.@nexprs $(2O) j -> result = result .+ interior_stencil[j] .* interior[j]
            Base.@nexprs i j -> result = result .+ exterior_stencil[j] .* exterior[j]

            result
        end
    end
end

# @generated function _prolong_interface(f::F, values::AbstractVector{T}, ctx::TransferContext{N, T, O, G}, level::Int, node::Int, cell::CartesianIndex{N}, ::Val{I}, rest::Vararg{Val, M}) where {N, T, O, I, M, G, F <: Function}
#     axis = M + 1
#     side = I > 0
#     face = FaceIndex{N}(axis, side)

#     # We are not computing along this axis
#     Gnew = setindex(G, 0, axis)

#     quote
#         ctxnew = TransferContext{$O}(ctx.basis, ctx.mesh, ctx.dofs, Val($Gnew))

        
#     end
# end

#######################
## Tuple Helpers ######
#######################

# Flatten a nested tuple
tuple_flatten(t::NTuple{N, T}) where {N, T} = t

@generated tuple_flatten(t::NTuple{N, NTuple{M, T}}) where {N, M, T} = :(
    Base.@ntuple $(N*M) index -> begin
        i = (index - 1) ÷ N + 1
        j = (index - 1) % M + 1
        t[i][j]
    end
)