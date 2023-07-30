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

"""
A god awful monstrosity of a function which computes target values for each exterior cell.
"""
@generated function compute_block_boundary(block::TransferBlock{N, T, O, I}, cell::CartesianIndex{N}) where {N, T, O, I}
    size = ntuple(i -> ifelse(I[i] ≠ 0, O, 1), N)
    len = prod(size)

    quote
        boundary = _compute_block_boundary(block, cell.I)
        SArray{Tuple{$(size...)}, $T, $N, $len}(boundary)
    end
end

@generated function _compute_block_boundary(block::TransferBlock{N, T, O, G, I}, cell::NTuple{N, Int}) where {N, T, O, G, I}
    axis = length(I)

    if axis == 0
        return quote
            # The base case in the recursion
            cells = nodecells(block.mesh, block.level)
            offset = nodeoffset(block.dofs, block.level, block.node)
            linear = LinearIndices(cells)
            return block.values[offset + linear[cell...]]
        end
    end

    side = I[axis] > 0

    if I[axis] == 0
        return :(_compute_block_boundary(TransferBlock(block), cell))
    end
    
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

            return tuple_flatten(_physical_boundary(condition, block, cell))
        elseif neighbor == 0
            # Neighbor is coarser
            return tuple_flatten(_coarse_interface(block, cell))
        elseif nodechildren(block.mesh, block.level, neighbor) == -1
            # Neighbor is same level
            return tuple_flatten(_identity_interface(block, cell))
        else
            # Neighbor is more refined
            return tuple_flatten(_refined_interface(block, cell))
        end

    end
end

@generated function _physical_boundary(condition::BoundaryCondition{T}, block::TransferBlock{N, T, O, G, I}, cell::NTuple{N, Int}) where {N, T, O, G, I}
    axis = length(I)
    side = I[axis] > 0

    quote
        total = nodecells(block.mesh, block.level)[$axis]
        width = nodebounds(block.mesh, block.level, block.node).widths[$axis]

        ccell = setindex(cell, $side ? total : 1, $axis)
        center = _compute_block_boundary(TransferBlock(block), ccell)

        interior = Base.@ntuple $(2O) j -> begin
            icell_j = setindex(cell, $side ? total - j : 1 + j, $axis)
            _compute_block_boundary(TransferBlock(block), icell_j)
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
            normal_to_global = $(I[axis]) * total / width

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

@generated function _identity_interface(block::TransferBlock{N, T, O, G, I}, cell::NTuple{N, Int}) where {N, T, O, I, G}
    axis = length(I)
    side = I[axis] > 0
    face = FaceIndex{N}(axis, side)
    
    quote
        neighbor = nodeneighbors(block.mesh, block.level, block.node)[$face]
        total = nodecells(block.mesh, block.level)[$axis]

        ccell = setindex(cell, $side ? total : 1, $axis)
        center = _compute_block_boundary(TransferBlock(block), ccell)

        interior = Base.@ntuple $(2O) j -> begin
            icell_j = setindex(cell, $side ? total - j : 1 + j, $axis)
            _compute_block_boundary(TransferBlock(block), icell_j)
        end

        nblock = setblocknode(block, block.level, neighbor)

        exterior = Base.@ntuple $(O) j -> begin
            ecell_j = setindex(cell, $side ? j : total + 1 - j, $axis)
            _compute_block_boundary(TransferBlock(nblock), ecell_j)
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

@generated function _prolong_interface(block::TransferBlock{N, T, O, G, I}, cell::NTuple{N, Int}, nlevel::Int, nnode::Int, npoint::NTuple{N, PointIndex}, ::Val{S}) where {N, T, O, I, G, S}
    axis = length(I)
    side = I[axis] > 0

    quote
        total = nodecells(block.mesh, nlevel)[$axis]

        # Fill interior of this cell
        ccell = setindex(cell, $side ? total : 1, $axis)
        center = _compute_block_boundary(TransferBlock(block), ccell)

        interior = Base.@ntuple $(2O) j -> begin
            icell_j = setindex(cell, $side ? total - j : 1 + j, $axis)
            _compute_block_boundary(TransferBlock(block), icell_j)
        end

        # Fill interior of child block
        nblock = setblocknode(block, nlevel, nnode)

        ncpoint = setindex(npoint, CellIndex($side ? 1 : total), $axis)
        ncenter = block_prolong(TransferBlock(nblock), ncpoint, nblock.basis)

        ninterior = Base.@ntuple $(2O) j -> begin
            nipoint_j = setindex(npoint, CellIndex($side ? 1 + j : total - j), $axis)
            block_prolong(TransferBlock(nblock), nipoint_j, nblock.basis)
        end

        # Solve for unknowns
        Base.@nexprs $O i -> begin
            vstencil_i = Stencil(block.basis, $side ? VertexValue{$(2O + 1), i, false}() : VertexValue{i, $(2O + 1), true}())
            dstencil_i = Stencil(block.basis, $side ? VertexDerivative{$(2O + 1), i, false}() : VertexDerivative{i, $(2O + 1), true}())

            nvstencil_i = Stencil(nblock.basis, $(!side) ? VertexValue{$(2O + 1), i, false}() : VertexValue{i, $(2O + 1), true}())
            ndstencil_i = Stencil(nblock.basis, $(!side) ? VertexDerivative{$(2O + 1), i, false}() : VertexDerivative{i, $(2O + 1), true}())

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
            interior_nvstencil_i = $(!side) ? nvstencil_i.left : nvstencil_i.right
            exterior_nvstencil_i = $(!side) ? nvstencil_i.right : nvstencil_i.left
            interior_ndstencil_i = $(!side) ? ndstencil_i.left : ndstencil_i.right
            exterior_ndstencil_i = $(!side) ? ndstencil_i.right : ndstencil_i.left

            nvresult_i = nvstencil_i.center .* ncenter
            ndresult_i = ndstencil_i.center .* ncenter

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
            srhs_i = SVector(SVector(nvresult_i .- vresult_i), SVector(ndresult_i .* S .- dresult_i))
            sx_i = smatrix_i \ srhs_i

            # Update unknowns
            unknown_i = Tuple(sx_i[1])
            nunknown_i = Tuple(sx_i[2])

            vapplied_i = vresult_i .+ exterior_vstencil_i[end] .* unknown_i
        end

        return Base.@ntuple $O i -> vapplied_i
    end
end

@generated function _refined_interface(block::TransferBlock{N, T, O, G, I}, cell::NTuple{N, Int}) where {N, T, O, G, I}
    axis = length(I)
    side = I[axis] > 0
    face = FaceIndex{N}(axis, side)

    quote
        cells = nodecells(block.mesh, block.level)
        neighbor = nodeneighbors(block.mesh, block.level, block.node)[$face]

        # Determine refined point
        halfcells = cells ./ 2

        # Find corresponding child of neighbor
        child = SplitIndex(setindex(cell .> halfcells, !$side, $axis))
        
        refinednode = nodechildren(block.mesh, block.level, neighbor) + child.linear

        refinedvertex = 2 .* (cell .- Tuple(child) .* halfcells)
        refinedpoint = Base.@ntuple $N i -> ifelse(i == $axis, CellIndex(0), VertexIndex(refinedvertex[i]))

        _prolong_interface(block, cell, block.level + 1, refinednode, refinedpoint, Val(1//2))
    end
end

@generated function _coarse_interface(block::TransferBlock{N, T, O, G, I}, cell::NTuple{N, Int}) where {N, T, O, G, I}
    axis = length(I)
    side = I[axis] > 0
    face = FaceIndex{N}(axis, side)

    quote
        cells = nodecells(block.mesh, block.level)

        parent = nodeparent(block.mesh, block.level, block.node)
        children = nodechildren(block.mesh, block.level - 1, parent)
        child = SplitIndex{$N}(block.node - children)
        coarsenode = nodeneighbors(block.mesh, block.level - 1, parent)[$face]

        coarsesubcell = cell .+ cells .* Tuple(child)
        # coarsepoint = Base.@ntuple $N i -> begin 
        #     if $(G)[i] == 1
        #         return CellIndex(1)
        #     elseif $(G)[i] == -1
        #         return CellIndex(cells[i])
        #     else
        #         return SubCellIndex(coarsesubcell[i])
        #     end
        # end

        coarsepoint = Base.@ntuple $N i -> ifelse(i == $axis, CellIndex(0), SubCellIndex(coarsesubcell[i]))

        _prolong_interface(block, cell, block.level - 1, coarsenode, coarsepoint, Val(2))
    end
end

#######################
## Tuple Helpers ######
#######################

# Flatten a nested tuple
tuple_flatten(t::NTuple{N, T}) where {N, T} = t

@generated tuple_flatten(t::NTuple{N, NTuple{M, T}}) where {N, M, T} = :(
    Base.@ntuple $(N*M) index -> begin
        i = (index - 1) ÷ $N + 1
        j = (index - 1) % $M + 1
        t[i][j]
    end
)