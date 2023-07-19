########################
## Block ###############
########################

export Block, blocktotal, blockcells, blockvalue, setblockvalue!
export cellindices, cellwidths, cellposition

struct Block{N, T, O}
    values::Array{T, N}

    Block{O}(values::Array{T, N}) where {N, T, O} = new{N, T, O}(values)
end

Block{N, T, O}(::UndefInitializer) where {N, T, O} = new{N, T, O}(Array{T, N}(undef, ntuple(i -> 2O + 1, Val(N))))

Base.size(block::Block) = size(block.values)
Base.fill!(block::Block{N, T}, v::T) where {N, T} = fill!(block.values, v)

blocktotal(block::Block) = size(block.values)
blockcells(block::Block{N, T, O}) where {N, T, O} = size(block.values) .- 2O
blockvalue(block::Block{N, T, O}, cell::CartesianIndex{N}) where {N, T, O} = block.value[CartesianIndex(cell.I .+ O)]
setblockvalue!(block::Block{N, T, O}, value::T, cell::CartesianIndex{N}) where {N, T, O} = block.values[CartesianIndex(cell.I .+ O)] = value

cellindices(block::Block) = CartesianIndex(blockcells(block))
cellwidths(block::Block{N, T}) where {N, T} = SVector{N, T}(1 ./ blockcells(block))
cellposition(block::Block{N, T}, cell::CartesianIndex{N}) where {N, T} = SVector{N, T}((cell.I .- T(1//2)) ./ blockcells(block))

#########################
## Stencil Product ######
#########################

export block_stencil_product

"""
Apply the tensor product of a set of stencils at a point on a numerical domain
"""
function block_stencil_product(block::Block{N, T}, cell::CartesianIndex{N}, stencils::NTuple{N, Stencil{T}}) where {N, T}
    _block_stencil_product(block, cell, stencils)
end

function _block_stencil_product(block::Block{N, T}, cell::CartesianIndex{N}, ::NTuple{0, Stencil{T}}) where {N, T}
    blockvalue(block, cell)
end

function _block_stencil_product(block::Block{N, T}, cell::CartesianIndex{N}, stencils::NTuple{L, Stencil{T}}) where {N, T, L}
    remaining = ntuple(i -> stencils[i], Val(L - 1))

    result = stencils[L].center * _block_stencil_product(block, cell, remaining)

    for (i, left) in enumerate(stencils[L].left)
        offcell = CartesianIndex(setindex(cell, cell[L] - i, L))
        result += left * _block_stencil_product(block, offcell, remaining)
    end

    for (i, right) in enumerate(stencils[L].right)
        offcell = CartesianIndex(setindex(cell, cell[L] + i, L))
        result += right * _block_stencil_product(block, offcell, remaining)
    end

    result
end

#######################
## Derivatives ########
#######################

export block_gradient, block_hessian

"""
Computes the gradient at a cell on a domain.
"""
@generated function block_gradient(block::Block{N, T, O}, cell::CartesianIndex{N}, basis::AbstractBasis{T}) where {N, T, O}
    quote 
        cells = blockcells(block)

        grad = Base.@ntuple $N i -> begin
            stencils = Base.@ntuple $N dim -> ifelse(i == dim, value_stencil(basis, Val(O), Val(1)), value_stencil(basis, Val(O), Val(0)))
            block_stencil_product(block, cell, stencils) * cells[i]
        end

        SVector{$N, $T}(grad) 
    end
end

"""
Computes the hessian at a cell on a domain.
"""
@generated function block_hessian(block::Block{N, T, O}, cell::CartesianIndex{N}, basis::AbstractBasis{T}) where {N, T, O}
    quote
        cells = blockcells(block)

        hess = Base.@ntuple $(N*N) index -> begin
            i = (index - 1) ÷ $N + 1
            j = (index - 1) % $N + 1
            if i == j
                stencils = Base.@ntuple $N dim -> ifelse(i == dim, value_stencil(basis, Val(O), Val(2)), value_stencil(basis, Val(O), Val(0)))
                result = block_stencil_product(block, cell, stencils) * cells[i]^2
            else
                stencils = Base.@ntuple $N dim -> ifelse(i == dim || j == dim, value_stencil(basis, Val(O), Val(1)), value_stencil(basis, Val(O), Val(0)))
                result = block_stencil_product(block, cell, stencils) * cells[i] * cells[j]
            end

            result
        end
        
        SMatrix{$N, $N, $T}(hess)
    end
end

#######################
## Diagonals ##########
#######################

export value_diagonal, gradient_diagonal, hessian_diagonal

function value_diagonal(::Val{N}, ::AbstractBasis{T}) where {N, T}
    one(T)
end

@generated function gradient_diagonal(::Val{N}, basis::AbstractBasis{T}) where {N, T}
    quote 
        cells = blockcells(block)

        grad = Base.@ntuple $N i -> begin
            stencils = Base.@ntuple $N dim -> ifelse(i == dim, value_stencil(basis, Val(O), Val(1)), value_stencil(basis, Val(O), Val(0)))
            stencil_diagonal(stencils) * cells[i]
        end

        SVector{$N, $T}(grad) 
    end
end

@generated function hessian_diagonal(::Val{N}, basis::AbstractBasis{T}) where {N, T}
    quote
        cells = blockcells(block)

        hess = Base.@ntuple $(N*N) index -> begin
            i = (index - 1) ÷ $N + 1
            j = (index - 1) % $N + 1
            if i == j
                stencils = Base.@ntuple $N dim -> ifelse(i == dim, value_stencil(basis, Val(O), Val(2)), value_stencil(basis, Val(O), Val(0)))
                result = stencil_diagonal(stencils) * cells[i]^2
            else
                stencils = Base.@ntuple $N dim -> ifelse(i == dim || j == dim, value_stencil(basis, Val(O), Val(1)), value_stencil(basis, Val(O), Val(0)))
                result = stencil_diagonal(stencils) * cells[i] * cells[j]
            end

            result
        end
        
        SMatrix{$N, $N, $T}(hess)
    end
end

#######################
## Prolongation #######
#######################

export block_prolong, block_prolong_interior

"""
Performs prolongation for a full domain.
"""
function block_prolong(block::Block{N, T, O}, point::NTuple{N, PointIndex}, basis::AbstractBasis{T}) where {N, T, O}
    cell = CartesianIndex(map(point_to_cell, point))
    stencils = map(i -> _point_to_prolong_stencil(i, basis, Val(O)), point) 
    block_stencil_product(block, cell, stencils)
end

function _point_to_prolong_stencil(::CellIndex, basis::AbstractBasis{T}, ::Val{O}) where {T, O}
    cell_value_stencil(basis, Val(0), Val(0))
end

function _point_to_prolong_stencil(::VertexIndex, basis::AbstractBasis{T}, ::Val{O}) where {T, O}
    vertex_value_stencil(basis, Val(O + 1), Val(O + 1), Val(false))
end

function _point_to_prolong_stencil(index::SubCellIndex, basis::AbstractBasis{T}, ::Val{O}) where {T, O}
    if subcell_side(index)
        return subcell_value_stencil(basis, Val(O), Val(O), Val(true))
    else
        return subcell_value_stencil(basis, Val(O), Val(O), Val(false))
    end
end

"""
Performs prolongation within a block, to the given order.
"""
function block_prolong_interior(block::Block{N, T, O}, point::NTuple{N, PointIndex}, basis::AbstractBasis{T}) where {N, T, O}
    cells = blockcells(block)
    cell = CartesianIndex(map(point_to_cell, point))
    stencils = map(i -> _point_to_prolong_stencil_interior(i, cells[i], basis, Val(O)), point) 
    block_stencil_product(block, cell, stencils)
end

function _point_to_prolong_stencil_interior(::CellIndex, ::Int, basis::AbstractBasis{T}, ::Val{O}) where {T, O}
    cell_value_stencil(basis, Val(0), Val(0))
end

@generated function _point_to_prolong_stencil_interior(index::VertexIndex, total::Int, basis::AbstractBasis{T}, ::Val{O}) where {T, O}
    quote 
        cindex = point_to_cell(index)

        leftcells = min($O, cindex)
        rightcells = min($O, total - cindex)

        # Left side
        if leftcells < $O
            Base.@nexprs $O i -> begin
                if leftcells == i - 1
                    return vertex_value_stencil(basis, Val(i - 1), Val($(2O + 1)), Val(false))
                end
            end
        end

        # Right side
        if rightcells < $O
            Base.@nexprs $O i -> begin
                if rightcells == i - 1
                    return vertex_value_stencil(basis, Val($(2O + 1)), Val(i - 1), Val(false))
                end
            end
        end

        return vertex_value_stencil(basis, Val($(O + 1)), Val($(O + 1)), Val(false))
    end
end

@generated function _point_to_prolong_stencil_interior(index::SubCellIndex, total::Int, basis::AbstractBasis{T}, ::Val{O}) where {T, O}
    side_expr = side -> quote 
        cindex = point_to_cell(index)

        leftcells = min($O, cindex - 1)
        rightcells = min($O, total - cindex)

        # Left side
        if leftcells < $O
            Base.@nexprs $O i -> begin
                if leftcells == i - 1
                    return subcell_value_stencil(basis, Val(i - 1), Val($(2O)), Val($side))
                end
            end
        end

        # Right side
        if rightcells < $O
            Base.@nexprs $O i -> begin
                if rightcells == i - 1
                    return subcell_value_stencil(basis, Val($(2O), Val(i - 1)), Val($side))
                end
            end
        end

        return subcell_value_stencil(basis, Val($(O)), Val($(O)), Val($side))
    end

    quote 
        if subcell_side(index)
            $(side_expr(true))
        else
            $(side_expr(false))
        end
    end
end

#############################
## Transfer #################
#############################

export fill_interior!, fill_interior_from_linear!
export block_surfaces, block_boundary

"""
Fills the interior of a block by calling `f` once for each interior cell (where the argument is a cartesian cell index).
"""
function fill_interior!(f::F, block::Block) where {F <: Function}
    for cell in cellindices(block)
        setblockvalue!(block, f(cell), cell)
    end
end

"""
Fills the interior of a block by calling `f` once for each interior cell (where the argument is a linear cell index).
"""
function fill_interior_from_linear!(f::F, block::Block) where {F <: Function}
    interior = blockinterior(block)
    linear = LinearIndices(interior)

    for cell in CartesianIndices(interior)
        setblockvalue!(block, f(linear[cell]), cell)
    end
end

"""
Iterates the subsurfaces of an `N`- dimensional block. The argument `Val(I)` is passed to `f`, where
`I` is an N-tuple of integers. 
"""
@generated function block_surfaces(f::F, ::Val{N}) where {N, F <: Function}
    # For a given i, find all subdomains with that number of edges.
    subdomain_exprs = i -> begin
        sub_exprs = Expr[]
        
        for subdomain in CartesianIndices(ntuple(_ -> 3, Val(N)))
            if sum(subdomain.I .== 1 .|| subdomain.I .== 3) == i
                push!(sub_exprs, :(f(Val($(subdomain.I .- 2)))))
            end
        end

        sub_exprs
    end

    # Build the final set of exprs
    exprs = Expr[]

    for i in 1:O
        append!(exprs, subdomain_exprs(i))
    end

    quote
        $(exprs...)
    end
end

"""
Calls `f` for each boundary cell.
"""
@generated function block_boundary(f::F, block::Block{N, T, O}) where {N, T, O, F <: Function}
    @inline block_surfaces(i -> _block_boundary(f, block, i), Val(N))
end

@generated function _block_boundary(f::F, block::Block{N, T, O}, ::Val{I}) where {N, T, O, I, F <: Function}
    facecells_exprs = ntuple(i -> ifelse(I[i] == 0, :(1:cells[$i]), :(1:1)), Val(N))
    facecell_exprs = ntuple(Val(N)) do i
        if I[i] == 1
            :(cells[$i])
        elseif I[i] == -1
            :(1)
        else
            :(facecell[$i])
        end
    end

    quote
        cells = blockcells(block)

        for facecell in CartesianIndices(tuple($(facecells_exprs...)))
            cell = CartesianIndex(tuple($(facecell_exprs...)))
            f(block, cell, Val($I))
        end
    end
end