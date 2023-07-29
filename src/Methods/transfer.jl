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

setblocknode(block::TransferBlock{N, T, O}, level::Int, node::Int, ::Val{I}) where {N, T, O, I} = TransferBlock{O}(block.f, block.values, block.basis, block.mesh, block.dofs, level, node, Val(I))

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
            return tuple_flatten(_prolong_interface(block, cell, Val($face)))
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

@generated function _identity_interface(block::TransferBlock{N, T, O, G}, cell::NTuple{N, Int}, ::Val{I}, rest::Vararg{Val, M}) where {N, T, O, I, M, G}
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
            _compute_block_boundary(setblocknode(block, block.level, neighbor, Val(G)), ecell_j, rest...)
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

@generated function _prolong_interface(block::TransferBlock{N, T, O, I}, cell::NTuple{N, Int}, ::Val{Face}) where {N, T, O, I, Face}
    axis = faceaxis(Face)
    side = faceside(Face)

    I_rest = ntuple(i -> ifelse(i < axis, I[axis], 0), N)

    quote
        cells = nodecells(block.mesh, block.level)
        neighbor = nodeneighbors(block.mesh, block.level, block.node)[$Face]
        
        total = cells[$axis]

        # Determine refined point

        halfcells = cells ./ 2

        child = cell .> halfcells
        # Set child to appropriate side
        child = setindex(child, !$side, $axis)

        refinednode = nodechildren(block.mesh, block.level, block.node) + child.linear
        refinedlevel = block.level

        refinedvertex = 2 .* (cell .- child .* halfcells)
        refinedpoint = Base.@ntuple $N i -> begin
            if i == $axis
                # Fill with invalid data currently
                return CellIndex(0)
            else
                return VertexIndex(refinedvertex[i])
            end
        end

        # Fill interior of this cell
        mblock = setblocknode(block, block.level, block.node, Val($I_rest))

        ccell = CartesianIndex(setindex(cell, $side ? total : 1, $axis))
        center = compute_block_boundary(mblock, cell)

        interior = Base.@ntuple $(2O) j -> begin
            icell_j = CartesianIndex(setindex(cell, $side ? total - j : 1 + j, $axis))
            compute_block_boundary(mblock, icell_j)
        end

        # Fill interior of child block
        rblock = setblocknode(block, refinedlevel, refinednode, Val($I_rest))

        rcpoint = setindex(refinedpoint, $side ? 1 : total, $axis)
        rcenter = block_prolong(rblock, rcpoint, rblock.basis)

        rinterior = Base.@ntuple $(2O) j -> begin
            ripoint_j = CartesianIndex(setindex(refinedpoint, $side ? 1 + j : total - j, $axis))
            block_prolong(rblock, ripoint_j, rblock.basis)
        end

        # Solve for unknowns

        Base.@nexprs $O i -> begin
            vstencil_i = Stencil(rblock.basis, $side ? VertexValue{i, $(2O + 1), false}() : VertexValue{$(2O + 1), i, true}())
            dstencil_i = Stencil(rblock.basis, $side ? VertexDerivative{i, $(2O + 1), false}() : VertexDerivative{$(2O + 1), i, true}())

            rvstencil_i = Stencil(rblock.basis, $!side ? VertexValue{i, $(2O + 1), false}() : VertexValue{$(2O + 1), i, true}())
            rdstencil_i = Stencil(rblock.basis, $!side ? VertexDerivative{i, $(2O + 1), false}() : VertexDerivative{$(2O + 1), i, true}())

            # This cell
            interior_vstencil_i = $side ? vstencil_i.left : vstencil_i.right
            exterior_vstencil_i = $side ? vstencil_i.right : vstencil_i.left
            interior_dstencil_i = $side ? dstencil_i.left : dstencil_i.right
            exterior_dstencil_i = $side ? dstencil_i.right : dstencil_i.left

            vresult_i = vstencil_i.center .* center
            dresult_i = dstencil_i.center .* center

            Base.@nexprs $(2O) j -> begin 
                vresult_i = vresult_i .+ interior_vstencil_i[j] .* interior[j]
                dresult_i = dresult_i .+ interior_dstencil_i[j] .* interior[j]
            end

            Base.@nexprs (i - 1) j -> begin
                vresult_i = vresult_i .+ exterior_vstencil_i[j] .* unknown_j
                dresult_i = dresult_i .+ exterior_dstencil_i[j] .* unknown_j
            end

            # Refined cell
            interior_rvstencil_i = $side ? rvstencil_i.left : rstencil_i.right
            exterior_rvstencil_i = $side ? rvstencil_i.right : rstencil_i.left
            interior_rdstencil_i = $side ? rdstencil_i.left : rstencil_i.right
            exterior_rdstencil_i = $side ? rdstencil_i.right : rstencil_i.left

            rvresult_i = rvstencil_i.center .* rcenter
            rdresult_i = rdstencil_i.center .* rcenter

            Base.@nexprs $(2O) j -> begin 
                rvresult_i = rvresult_i .+ interior_rvstencil_i[j] .* rinterior[j]
                rdresult_i = rdresult_i .+ interior_rdstencil_i[j] .* rinterior[j]
            end

            Base.@nexprs (i - 1) j -> begin
                rvresult_i = rvresult_i .+ rexterior_vstencil_i[j] .* runknown_j
                rdresult_i = rdresult_i .+ rexterior_dstencil_i[j] .* runknown_j
            end

            # Solve
            smatrix_i = 
                SA[
                    exterior_vstencil_i[end] (-exterior_rvstencil_i[end]);
                    exterior_dstencil_i[end] (-exterior_rdstencil_i[end] / 2)
                ]
            srhs_i = SVector(SVector(rvresult_i .- vresult_i), SVector(rdresult_i  ./ 2 .- dresult_i))
            sx_i = smatrix_i \ srhs_i

            # Update unknowns
            unknown_i = Tuple(sx_i[1])
            runknown_i = Tuple(sx_i[2])

            vapplied_i = vresult_i .+ exterior_vstencil_i[end] .* unknown_i
        end

        return Base.@ntuple $O i -> vapplied_i
    end
end

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