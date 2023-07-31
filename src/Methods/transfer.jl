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
        compute_block_boundary(TransferBlock{O}(f, values, basis, mesh, dofs, level, node, i), cell)
    end
end

struct TransferBlock{N, T, O, G, I, F, V, B} <: AbstractBlock{N, T, O} 
    values::V
    f::F
    basis::B
    mesh::Mesh{N, T}
    dofs::DoFManager{N, T}
    level::Int
    node::Int

    TransferBlock{O, G, I}(f::Function, values::AbstractVector{T}, basis::AbstractBasis{T}, mesh::Mesh{N, T}, dofs::DoFManager{N, T}, level::Int, node::Int) where {N, T, O, G, I} = new{N, T, O, G, I, typeof(f), typeof(values), typeof(basis)}(values, f, basis, mesh, dofs, level, node)
end

TransferBlock{O}(f::Function, values::AbstractVector{T}, basis::AbstractBasis{T}, mesh::Mesh{N, T}, dofs::DoFManager{N, T}, level::Int, node::Int, ::Val{G}) where {N, T, O, G} = TransferBlock{O, G, G}(f, values, basis, mesh, dofs, level, node)
TransferBlock(block::TransferBlock{N, T, O, G, I}) where {N, T, O, G, I} = TransferBlock{O, G, Base.front(I)}(block.f, block.values, block.basis, block.mesh, block.dofs, block.level, block.node)

setblocknode(block::TransferBlock{N, T, O, G, I}, level::Int, node::Int) where {N, T, O, G, I} = TransferBlock{O, G, I}(block.f, block.values, block.basis, block.mesh, block.dofs, level, node)

Blocks.blocktotal(block::TransferBlock) = nodecells(block.mesh, block.level)
Blocks.blockcells(block::TransferBlock) = nodecells(block.mesh, block.level)
Blocks.blockvalue(block::TransferBlock{N}, cell::CartesianIndex{N}) where N = _compute_block_boundary(block, cell.I)

count_tuple_length(O, I) = length(I) == 0 ? 1 : prod(ntuple(i -> ifelse(I[i] ≠ 0, O, 1), length(I)))

"""
A god awful monstrosity of a function which computes target values for each exterior cell.
"""
@generated function compute_block_boundary(block::TransferBlock{N, T, O, G}, cell::CartesianIndex{N}) where {N, T, O, G}
    size = ntuple(i -> ifelse(G[i] ≠ 0, O, 1), N)
    len = count_tuple_length(O, G)

    quote
        boundary = _compute_block_boundary(block, cell.I)
        SArray{Tuple{$(size...)}, $T, $N, $len}(boundary)
    end
end

@generated function _compute_block_boundary(block::TransferBlock{N, T, O, G, I}, cell::NTuple{N, Int}) where {N, T, O, G, I}
    axis = length(I)
    len = count_tuple_length(O, I)

    if axis == 0
        return quote
            # The base case in the recursion
            cells = nodecells(block.mesh, block.level)
            offset = nodeoffset(block.dofs, block.level, block.node)
            linear = LinearIndices(cells)
            return tuple(block.values[offset + linear[cell...]])::NTuple{$len, $T}
        end
    end

    side = I[axis] > 0

    if I[axis] == 0
        return :(_compute_block_boundary(TransferBlock(block), cell)::NTuple{$len, $T})
    end
    
    face = FaceIndex{N}(axis, side)

    quote
        # Cache fields for some reason
        mesh::Mesh{$N, $T, $(2N)} = block.mesh
        dofs::DoFManager{$N, $T} = block.dofs
        level = block.level
        node = block.node

        cells = nodecells(mesh, level)
        neighbor = nodeneighbors(mesh, level, node)[$(face)]
        transform = nodetransform(mesh, level, node)
        position = cellposition(mesh, level, CartesianIndex(cell))
        offset = nodeoffset(dofs, level, node)

        if neighbor < 0
            gpos = transform(position .+ G .* 1 ./ (2 .* cells))
            condition = block.f(gpos, $face)

            return tuple_flatten(_physical_boundary(condition, block, cell))::NTuple{$len, $T}
        elseif neighbor == 0
            # Neighbor is coarser
            return tuple_flatten(_coarse_interface(block, cell))::NTuple{$len, $T}
        elseif nodechildren(mesh, level, neighbor) == -1
            # Neighbor is same level
            return tuple_flatten(_identity_interface(block, cell))::NTuple{$len, $T}
        else
            # Neighbor is more refined
            return tuple_flatten(_refined_interface(block, cell))::NTuple{$len, $T}
        end

    end
end

@generated function _physical_boundary(condition::BoundaryCondition{T}, block::TransferBlock{N, T, O, G, I}, cell::NTuple{N, Int}) where {N, T, O, G, I}
    axis = length(I)
    side = I[axis] > 0

    rlen = count_tuple_length(O, Base.front(I))

    quote
        mesh::Mesh{$N, $T, $(2N)} = block.mesh
        level::Int = block.level
        node::Int = block.node

        total::Int = nodecells(mesh, level)[$axis]
        width::$T = nodebounds(mesh, level, node).widths[$axis]

        ccell::NTuple{$N, Int} = setindex(cell, $side ? total : 1, $axis)
        center::NTuple{$rlen, $T} = _compute_block_boundary(TransferBlock(block), ccell)

        interior::NTuple{$(2O), NTuple{$rlen, $T}} = Base.@ntuple $(2O) j -> begin
            icell_j = setindex(cell, $side ? total - j : 1 + j, $axis)
            _compute_block_boundary(TransferBlock(block), icell_j)
        end

        # Transform normal derivative
        normal_to_global::$T = $(I[axis]) * total / width

        Base.@nexprs $O i -> begin
            # Value
            vstencil_i = Stencil(block.basis, $side ? VertexValue{$(2O + 1), i, false}() : VertexValue{i, $(2O + 1), true}())
            
            vresult_i::NTuple{$rlen, $T} = vstencil_i.center .* center

            interior_vstencil_i::NTuple{$(2O), $T} = $side ? vstencil_i.left : vstencil_i.right
            exterior_vstencil_i::NTuple{i, $T} = $side ? vstencil_i.right : vstencil_i.left

            Base.@nexprs $(2O) j -> vresult_i = vresult_i .+ interior_vstencil_i[j] .* interior[j]
            Base.@nexprs (i - 1) j -> vresult_i = vresult_i .+ exterior_vstencil_i[j] .* unknown_j

            # Derivative
            dstencil_i = Stencil(block.basis, $side ? VertexDerivative{$(2O + 1), i, false}() : VertexDerivative{i, $(2O + 1), true}())
            
            dresult_i::NTuple{$rlen, $T} = dstencil_i.center .* center

            interior_dstencil_i::NTuple{$(2O), $T} = $side ? dstencil_i.left : dstencil_i.right
            exterior_dstencil_i::NTuple{i, $T} = $side ? dstencil_i.right : dstencil_i.left

            Base.@nexprs $(2O) j -> dresult_i = dresult_i .+ interior_dstencil_i[j] .* interior[j]
            Base.@nexprs (i - 1) j -> dresult_i = dresult_i .+ exterior_dstencil_i[j] .* unknown_j

            vnumerator_i::NTuple{$rlen, $T} = condition.value .* vresult_i
            dnumerator_i::NTuple{$rlen, $T} = condition.normal .* normal_to_global .* dresult_i
            rnumerator_i::$T = condition.rhs

            vdenominator_i::$T = condition.value * exterior_vstencil_i[end]
            ddenominator_i::$T = condition.normal * normal_to_global * exterior_dstencil_i[end] 

            unknown_i::NTuple{$rlen, $T} = (rnumerator_i .- vnumerator_i .- dnumerator_i) ./ (vdenominator_i + ddenominator_i)

            vapplied_i::NTuple{$rlen, $T} = vresult_i .+ exterior_vstencil_i[end] .* unknown_i
        end

        return Base.@ntuple $O i -> vapplied_i
    end
end

@generated function _identity_interface(block::TransferBlock{N, T, O, G, I}, cell::NTuple{N, Int}) where {N, T, O, I, G}
    axis = length(I)
    side = I[axis] > 0
    face = FaceIndex{N}(axis, side)

    rlen = count_tuple_length(O, Base.front(I))
    
    quote
        mesh::Mesh{$N, $T, $(2N)} = block.mesh
        level::Int = block.level
        node::Int = block.node

        neighbor::Int = nodeneighbors(mesh, level, node)[$face]
        total::Int = nodecells(mesh, level)[$axis]

        ccell = setindex(cell, $side ? total : 1, $axis)
        center::NTuple{$rlen, $T} = _compute_block_boundary(TransferBlock(block), ccell)

        interior::NTuple{$(2O), NTuple{$rlen, $T}} = Base.@ntuple $(2O) j -> begin
            icell_j = setindex(cell, $side ? total - j : 1 + j, $axis)
            _compute_block_boundary(TransferBlock(block), icell_j)
        end

        nblock = setblocknode(block, level, neighbor)

        exterior ::NTuple{$(O), NTuple{$rlen, $T}}= Base.@ntuple $(O) j -> begin
            ecell_j = setindex(cell, $side ? j : total + 1 - j, $axis)
            _compute_block_boundary(TransferBlock(nblock), ecell_j)
        end

        Base.@nexprs $O i -> begin
            stencil_i = Stencil(block.basis, $side ? VertexValue{$(2O + 1), i, false}() : VertexValue{i, $(2O + 1), true}())

            interior_stencil_i::NTuple{$(2O), $T} = $side ? stencil_i.left : stencil_i.right
            exterior_stencil_i::NTuple{i, $T} = $side ? stencil_i.right : stencil_i.left

            result_i::NTuple{$rlen, $T} = stencil_i.center .* center

            Base.@nexprs $(2O) j -> result_i = result_i .+ interior_stencil_i[j] .* interior[j]
            Base.@nexprs i j -> result_i = result_i .+ exterior_stencil_i[j] .* exterior[j]

            result_i
        end

        return Base.@ntuple $O i -> result_i
    end
end

@generated function _prolong_interface(block::TransferBlock{N, T, O, G, I}, cell::NTuple{N, Int}, nlevel::Int, nnode::Int, npoint::NTuple{N, PointIndex}, ::Val{S}) where {N, T, O, I, G, S}
    axis = length(I)
    side = I[axis] > 0

    rlen = count_tuple_length(O, Base.front(I))

    quote
        mesh::Mesh{$N, $T, $(2N)} = block.mesh
        level::Int = block.level
        node::Int = block.node
        
        total = nodecells(mesh, nlevel)[$axis]

        # Fill interior of this cell
        ccell::NTuple{$N, Int} = setindex(cell, $side ? total : 1, $axis)
        center::NTuple{$rlen, $T} = _compute_block_boundary(TransferBlock(block), ccell)

        interior::NTuple{$(2O), NTuple{$rlen, $T}} = Base.@ntuple $(2O) j -> begin
            icell_j = setindex(cell, $side ? total - j : 1 + j, $axis)
            _compute_block_boundary(TransferBlock(block), icell_j)
        end

        # Fill interior of child block
        nblock = setblocknode(block, nlevel, nnode)

        ncpoint::NTuple{$N, PointIndex} = setindex(npoint, CellIndex($side ? 1 : total), $axis)
        ncenter::NTuple{$rlen, $T} = block_prolong(TransferBlock(nblock), ncpoint, nblock.basis)

        ninterior::NTuple{$(2O), NTuple{$rlen, $T}} = Base.@ntuple $(2O) j -> begin
            nipoint_j = setindex(npoint, CellIndex($side ? 1 + j : total - j), $axis)
            block_prolong(TransferBlock(nblock), nipoint_j, nblock.basis)
        end

        # Solve for unknowns
        Base.@nexprs $O i -> begin
            # This cell
            vstencil_i = Stencil(block.basis, $side ? VertexValue{$(2O + 1), i, false}() : VertexValue{i, $(2O + 1), true}())
            dstencil_i = Stencil(block.basis, $side ? VertexDerivative{$(2O + 1), i, false}() : VertexDerivative{i, $(2O + 1), true}())

            interior_vstencil_i::NTuple{$(2O), $T} = $side ? vstencil_i.left : vstencil_i.right
            exterior_vstencil_i::NTuple{i, $T} = $side ? vstencil_i.right : vstencil_i.left
            interior_dstencil_i::NTuple{$(2O), $T} = $side ? dstencil_i.left : dstencil_i.right
            exterior_dstencil_i::NTuple{i, $T} = $side ? dstencil_i.right : dstencil_i.left

            vresult_i::NTuple{$rlen, $T} = vstencil_i.center .* center
            dresult_i::NTuple{$rlen, $T} = dstencil_i.center .* center

            Base.@nexprs $(2O) j -> begin 
                vresult_i = vresult_i .+ interior_vstencil_i[j] .* interior[j]
                dresult_i = dresult_i .+ interior_dstencil_i[j] .* interior[j]
            end

            Base.@nexprs (i - 1) j -> begin
                vresult_i = vresult_i .+ exterior_vstencil_i[j] .* unknown_j
                dresult_i = dresult_i .+ exterior_dstencil_i[j] .* unknown_j
            end

            # Refined cell
            nvstencil_i = Stencil(nblock.basis, $(!side) ? VertexValue{$(2O + 1), i, false}() : VertexValue{i, $(2O + 1), true}())
            ndstencil_i = Stencil(nblock.basis, $(!side) ? VertexDerivative{$(2O + 1), i, false}() : VertexDerivative{i, $(2O + 1), true}())

            interior_nvstencil_i::NTuple{$(2O), $T} = $(!side) ? nvstencil_i.left : nvstencil_i.right
            exterior_nvstencil_i::NTuple{i, $T} = $(!side) ? nvstencil_i.right : nvstencil_i.left
            interior_ndstencil_i::NTuple{$(2O), $T} = $(!side) ? ndstencil_i.left : ndstencil_i.right
            exterior_ndstencil_i::NTuple{i, $T} = $(!side) ? ndstencil_i.right : ndstencil_i.left

            nvresult_i::NTuple{$rlen, $T} = nvstencil_i.center .* ncenter
            ndresult_i::NTuple{$rlen, $T} = ndstencil_i.center .* ncenter

            Base.@nexprs $(2O) j -> begin 
                nvresult_i = nvresult_i .+ interior_nvstencil_i[j] .* ninterior[j]
                ndresult_i = ndresult_i .+ interior_ndstencil_i[j] .* ninterior[j]
            end

            Base.@nexprs (i - 1) j -> begin
                nvresult_i = nvresult_i .+ exterior_nvstencil_i[j] .* nunknown_j
                ndresult_i = ndresult_i .+ exterior_ndstencil_i[j] .* nunknown_j
            end

            # Solve
            smatrix_i = 
                SA[
                    exterior_vstencil_i[end] (-exterior_nvstencil_i[end]);
                    exterior_dstencil_i[end] (-exterior_ndstencil_i[end] * S)
                ]
            # Invert matrix
            smatrix_inv_i = inv(smatrix_i)
            
            srhs_i = SVector(SVector(nvresult_i .- vresult_i), SVector(ndresult_i .* S .- dresult_i))
            sx_i = smatrix_inv_i * srhs_i

            # Update unknowns
            unknown_i::NTuple{$rlen, $T} = Tuple(sx_i[1])
            nunknown_i::NTuple{$rlen, $T} = Tuple(sx_i[2])

            vapplied_i::NTuple{$rlen, $T} = vresult_i .+ exterior_vstencil_i[end] .* unknown_i
        end

        return Base.@ntuple $O i -> vapplied_i
    end
end

@generated function _refined_interface(block::TransferBlock{N, T, O, G, I}, cell::NTuple{N, Int}) where {N, T, O, G, I}
    axis = length(I)
    side = I[axis] > 0
    face = FaceIndex{N}(axis, side)

    quote
        # Cache fields for some reason
        mesh::Mesh{$N, $T, $(2N)} = block.mesh
        level::Int = block.level
        node::Int = block.node

        cells = nodecells(mesh, level)
        neighbor::Int = nodeneighbors(mesh, level, node)[$face]

        # Determine refined point
        halfcells::NTuple{$N, Int} = cells .÷ 2

        # Find corresponding child of neighbor
        child = SplitIndex(setindex(cell .> halfcells, !$side, $axis))
        
        refinednode::Int = nodechildren(mesh, level, neighbor) + child.linear

        refinedvertex::NTuple{$N, Int} = 2 .* (cell .- Tuple(child) .* halfcells)
        refinedpoint::NTuple{$N, PointIndex} = Base.@ntuple $N i -> begin 
            if i ≤ $axis
                ifelse($I[i] ≠ 0, ifelse($I[i] == 1, CellIndex(1), CellIndex(cells[i])), VertexIndex(refinedvertex[i]))
            else
                VertexIndex(refinedvertex[i])
            end
        end

        _prolong_interface(block, cell, level + 1, refinednode, refinedpoint, Val(1//2))
    end
end

@generated function _coarse_interface(block::TransferBlock{N, T, O, G, I}, cell::NTuple{N, Int}) where {N, T, O, G, I}
    axis = length(I)
    side = I[axis] > 0
    face = FaceIndex{N}(axis, side)

    quote
        # Cache fields for some reason (to avoid spurious allocations)
        mesh::Mesh{$N, $T, $(2N)} = block.mesh
        level::Int = block.level
        node::Int = block.node

        cells = nodecells(mesh, level)

        parent::Int = nodeparent(mesh, level, node)
        children::Int = nodechildren(mesh, level - 1, parent)
        child = SplitIndex{$N}(node - children)
        coarsenode::Int = nodeneighbors(mesh, level - 1, parent)[$face]
    
        coarsesubcell::NTuple{$N, Int} = cell .+ cells .* Tuple(child)
        coarsepoint::NTuple{$N, PointIndex} = Base.@ntuple $N i -> begin 
            if i ≤ $axis
                ifelse($I[i] ≠ 0, ifelse($I[i] == 1, CellIndex(1), CellIndex(cells[i])), SubCellIndex(coarsesubcell[i]))
            else
                SubCellIndex(coarsesubcell[i])
            end
        end

        _prolong_interface(block, cell, block.level - 1, coarsenode, coarsepoint, Val(2))
    end
end

#######################
## Tuple Helpers ######
#######################

# Flatten a nested tuple
# tuple_flatten(t::NTuple{N, T}) where {N, T} = t

@generated tuple_flatten(t::NTuple{N, NTuple{M, T}}) where {N, M, T} = :(
    Base.@ntuple $(N*M) index -> begin
        i = (index - 1) ÷ $N + 1
        j = (index - 1) % $M + 1
        t[i][j]
    end
)