############################
## Transfer ################
############################

export transfer_to_block!

"""
Transfers data from vector to block a block, with the given order of accuracy.
"""
function transfer_to_block!(f::F, block::AbstractBlock{N, T, O}, values::AbstractVector{T}, basis::AbstractBasis{T}, mesh::Mesh{N, T}, dofs::DoFManager{N, T}, level::Int, node::Int) where {N, T, O, F <: Function}
    # Fill interior 
    offset = nodeoffset(dofs, level, node)
    
    fill_interior_from_linear!(block) do i
        values[offset + i]
    end

    # Fill boundaries
    fill_boundaries!(block, basis) do cell, i
        @inline compute_boundary_for_block(f, values, TransferContext{O}(basis, mesh, dofs, i), level, node, cell)
    end
end

struct TransferContext{N, T, O, I, B, M, D}
    basis::B
    mesh::M
    dofs::D

    TransferContext{O}(basis::AbstractBasis{T}, mesh::Mesh{N, T}, dofs::DoFManager{N, T}, ::Val{I}) where {N, T, O, I} = new{N, T, O, I, typeof(basis), typeof(mesh), typeof(dofs)}(basis, mesh, dofs)
end

"""
A god awful monstrosity of a function which computes target values for each exterior cell.
"""
@generated function compute_boundary_for_block(f::F, values::AbstractVector{T}, ctx::TransferContext{N, T, O, I}, level::Int, node::Int, cell::CartesianIndex{N}) where {N, T, O, I, F <: Function}
    size = ntuple(i -> ifelse(I[i] ≠ 0, O, 1), N)
    len = prod(size)

    reversed::Tuple = map(Val, reverse(I))

    quote
        boundary = _compute_boundary_for_block(f, values, ctx, level, node, cell, $(reversed...))
        SArray{Tuple{$(size...)}, $T, $N, $len}(tuple_flatten(boundary))
    end
end

function _compute_boundary_for_block(f::F, values::AbstractVector{T}, ctx::TransferContext{N, T}, level::Int, node::Int, cell::CartesianIndex{N}) where {N, T, F <: Function}
    # The base case in the recursion
    cells = nodecells(ctx.mesh, level)
    offset = nodeoffset(ctx.dofs, level, node)
    linear = LinearIndices(cells)
    return values[offset + linear[cell]]
end

function _compute_boundary_for_block(f::F, values::AbstractVector{T}, ctx::TransferContext{N, T}, level::Int, node::Int, cell::CartesianIndex{N}, ::Val{0}, rest::Vararg{Val, M}) where {N, T, M, F <: Function}
    # This axis does not need to be extended
    _compute_boundary_for_block(f, values, ctx, level, node, cell, rest...)
end

@generated function _compute_boundary_for_block(f::F, values::AbstractVector{T}, ctx::TransferContext{N, T, O, G}, level::Int, node::Int, cell::CartesianIndex{N}, ::Val{I}, rest::Vararg{Val, M}) where {N, T, O, I, G, M, F <: Function}
    axis = M + 1
    side = I > 0
    face = FaceIndex{N}(axis, side)

    quote
        cells = nodecells(ctx.mesh, level)
        neighbor = nodeneighbors(ctx.mesh, level, node)[$face]
        transform = nodetransform(ctx.mesh, level, node)
        position = cellposition(ctx.mesh, level, cell)
        offset = nodeoffset(ctx.dofs, level, node)

        if neighbor < 0
            gpos = transform(position .+ G .* 1 ./ (2 .* cells))
            condition = f(gpos, $face)

            return _physical_boundary(f, values, condition, ctx, level, node, cell, Val(I), rest...)
        elseif neighbor == 0
            # Neighbor is coarser
            error("Coarse neighbor unimplemented")
        elseif nodechildren(ctx.mesh, level, neighbor) == -1
            # Neighbor is same level
            return _identity_interface(f, values, ctx, level, node, cell, Val(I), rest...)
        else
            # Neighbor is more refined
            error("Refined  neighbor unimplemented")
        end
    end
end

@generated function _physical_boundary(f::F, values::AbstractVector{T}, condition::BoundaryCondition{T}, ctx::TransferContext{N, T, O}, level::Int, node::Int, cell::CartesianIndex{N}, ::Val{I}, rest::Vararg{Val, M}) where {N, T, O, I, M, F <: Function}
    axis = M + 1
    side = I > 0
   
    quote
        total = nodecells(ctx.mesh, level)[$axis]
        width = nodebounds(ctx.mesh, level, node).widths[$axis]

        ccell = CartesianIndex(setindex(cell.I, $side ? total : 1, $axis))
        center = _compute_boundary_for_block(f, values, ctx, level, node, ccell, rest...)

        interior = Base.@ntuple $(2O) j -> begin
            icell_j = CartesianIndex(setindex(cell.I, $side ? total - j : 1 + j, $axis))
            _compute_boundary_for_block(f, values, ctx, level, node, icell_j, rest...)
        end

        Base.@nexprs $O i -> begin
            # Value
            vstencil_i = Stencil(ctx.basis, $side ? VertexValue{$(2O + 1), i, false}() : VertexValue{i, $(2O + 1), true}())
            vresult_i = tuple_mul(vstencil_i.center, center)

            interior_vstencil_i = $side ? vstencil_i.left : vstencil_i.right
            exterior_vstencil_i = $side ? vstencil_i.right : vstencil_i.left

            Base.@nexprs $(2O) j -> vresult_i = tuple_add(vresult_i, tuple_mul(interior_vstencil_i[j], interior[j]))
            Base.@nexprs (i - 1) j -> vresult_i = tuple_add(vresult_i, tuple_mul(exterior_vstencil_i[j], unknown_j))

            # Derivative
            dstencil_i = Stencil(ctx.basis, $side ? VertexDerivative{$(2O + 1), i, false}() : VertexDerivative{i, $(2O + 1), true}())
            dresult_i = tuple_mul(dstencil_i.center, center)

            interior_dstencil_i = $side ? dstencil_i.left : dstencil_i.right
            exterior_dstencil_i = $side ? dstencil_i.right : dstencil_i.left

            Base.@nexprs $(2O) j -> dresult_i = tuple_add(dresult_i, tuple_mul(interior_dstencil_i[j], interior[j]))
            Base.@nexprs (i - 1) j -> dresult_i = tuple_add(dresult_i, tuple_mul(exterior_dstencil_i[j], unknown_j))

            # Transform normal derivative
            normal_to_global = I * total / width

            vnumerator = tuple_mul(condition.value, vresult_i)
            dnumerator = tuple_mul(condition.normal, tuple_mul(normal_to_global, dresult_i))
            rnumerator = -condition.rhs

            vdenominator = condition.value * exterior_vstencil_i[end]
            ddenominator = condition.normal * normal_to_global * exterior_dstencil_i[end] 

            unknown_i = tuple_mul(-1/(vdenominator + ddenominator), tuple_add(tuple_add(vnumerator, dnumerator), rnumerator))

            vapplied_i = tuple_add(vresult_i, tuple_mul(exterior_vstencil_i[end], unknown_i))
        end

        return Base.@ntuple $O i -> vapplied_i
    end
end

@generated function _identity_interface(f::F, values::AbstractVector{T}, ctx::TransferContext{N, T, O}, level::Int, node::Int, cell::CartesianIndex{N}, ::Val{I}, rest::Vararg{Val, M}) where {N, T, O, I, M, F <: Function}
    axis = M + 1
    side = I > 0
    face = FaceIndex{N}(axis, side)
    
    quote
        neighbor = nodeneighbors(ctx.mesh, level, node)[$face]
        total = nodecells(ctx.mesh, level)[$axis]

        ccell = CartesianIndex(setindex(cell.I, $side ? total : 1, $axis))
        center = _compute_boundary_for_block(f, values, ctx, level, node, ccell, rest...)

        interior = Base.@ntuple $(2O) j -> begin
            icell_j = CartesianIndex(setindex(cell.I, $side ? total - j : 1 + j, $axis))
            _compute_boundary_for_block(f, values, ctx, level, node, icell_j, rest...)
        end

        exterior = Base.@ntuple $(O) j -> begin
            ecell_j = CartesianIndex(setindex(cell, $side ? j : total + 1 - j, $axis))
            _compute_boundary_for_block(f, values, ctx, level, neighbor, ecell_j, rest...)
        end

        return Base.@ntuple $O i -> begin
            stencil = Stencil(ctx.basis, $side ? VertexValue{$(2O + 1), i, false}() : VertexValue{i, $(2O + 1), true}())

            interior_stencil = $side ? stencil.left : stencil.right
            exterior_stencil = $side ? stencil.right : stencil.left

            result = tuple_mul(stencil.center, center)

            Base.@nexprs $(2O) j -> result = tuple_add(result, tuple_mul(interior_stencil[j], interior[j]))
            Base.@nexprs i j -> result = tuple_add(result, tuple_mul(exterior_stencil[j], exterior[j]))

            result
        end
    end
end

@generated function _prolong_interface(f::F, values::AbstractVector{T}, ctx::TransferContext{N, T, O, G}, level::Int, node::Int, cell::CartesianIndex{N}, ::Val{I}, rest::Vararg{Val, M}) where {N, T, O, I, M, G, F <: Function}
    axis = M + 1
    side = I > 0
    face = FaceIndex{N}(axis, side)

    # We are not computing along this axis
    Gnew = setindex(G, 0, axis)

    quote
        ctxnew = TransferContext{$O}(ctx.basis, ctx.mesh, ctx.dofs, Val($Gnew))

        
    end
end

#######################
## Stencil Product ####
#######################

function boundary_stencil_product(f::F, values::AbstractVector{T}, ctx::TransferContext{N, T}, level::Int, node::Int, cell::CartesianIndex{N}, stencils::NTuple{N, Stencil{T}}) where {N, T, F <: Function}
    _boundary_stencil_product(f, values, ctx, level, node, cell.I, reverse(stencils)...)
end

function _boundary_stencil_product(f::F, values::AbstractVector{T}, ctx::TransferContext{N, T}, level::Int, node::Int, cell::NTuple{N, Int}) where {N, T, F <: Function}
    _compute_boundary_for_block(f, values, ctx, level, node, CartesianIndex(cell))
end

function _boundary_stencil_product(f::F, values::AbstractVector{T}, ctx::TransferContext{N, T}, level::Int, node::Int, cell::NTuple{N, Int}, stencil::Stencil{T, L, R}, rest::Vararg{Stencil{T}, M}) where {N, T, L, R, M, F <: Function}
    axis = M + 1

    result = stencil.center * _block_stencil_product(block, cell, rest)

    result = tuple_mul(stencil.center, _boundary_stencil_product(f, values, ctx, level, node, cell, rest...))
        
    Base.@nexprs $L i -> begin
        lcell_i = setindex(cell, cell[axis] - i, axis)
        result = tuple_add(result,  _boundary_stencil_product(f, values, ctx, level, node, lcell_i, rest...))
    end

    Base.@nexprs $R i -> begin
        rcell_i = setindex(cell, cell[axis] + i, axis)
        result = tuple_add(result,  _boundary_stencil_product(f, values, ctx, level, node, rcell_i, rest...))
    end

    result
end

#######################
## Prolongation #######
#######################

function prolong_boundary_block(f::F, values::AbstractVector{T}, ctx::TransferContext{N, T, O}, level::Int, node::Int, point::NTuple{N, PointIndex}) where {N, T, F <: Function}
    cells = nodecells(ctx.mesh, level)
    cell = CartesianIndex(map(point_to_cell, point))
    stencils = ntuple(i ->  interior_prolong_stencil(point[i], cells[i], basis, Val(O)), Val(N))
    boundary_stencil_product(f, values, ctx, level, node, cell, stencils)
end

#######################
## Tuple Helpers ######
#######################

# Nested tuple multiplication
tuple_mul(t1::T, t2::T) where T = t1 * t2
tuple_mul(t1::T, t2::T) where T <: Tuple = tuple_mul.(t1, t2)
tuple_mul(v::T, t::Tuple) where T = tuple_mul.(v, t)
tuple_mul(t::Tuple, v::T) where T = tuple_mul.(v, t)

# Nested tuple addition
tuple_add(t1::T, t2::T) where T = t1 + t2
tuple_add(t1::T, t2::T) where T <: Tuple = tuple_add.(t1, t2)
tuple_add(v::T, t::Tuple) where T = tuple_add.(v, t)
tuple_add(t::Tuple, v::T) where T = tuple_add.(v, t)

# Flatten a nested tuple
tuple_flatten(t::NTuple{N, T}) where {N, T} = t

@generated tuple_flatten(t::NTuple{N, NTuple{M, T}}) where {N, M, T} = :(
    Base.@ntuple $(N*M) index -> begin
        i = (index - 1) ÷ N + 1
        j = (index - 1) % M + 1
        t[i][j]
    end
)