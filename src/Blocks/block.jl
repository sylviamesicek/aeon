########################
## Block ###############
########################

export Block, blocktotal, blockcells, blockvalue, setblockvalue!
export cellindices, cellwidths, cellposition

struct Block{N, T, O}
    values::Array{T, N}

    Block{O}(values::Array{T, N}) where {N, T, O} = new{N, T, O}(values)
end

Block{T, O}(::UndefInitializer, dims::Vararg{Int, N}) where {N, T, O} = Block{O}(Array{T, N}(undef, (dims .+ 2O)...))  
Block{N, T, O}(::UndefInitializer) where {N, T, O} = Block{T, O}(undef, ntuple(i -> 2O + 1, Val(N))...)

Base.size(block::Block) = size(block.values)
Base.fill!(block::Block{N, T}, v::T) where {N, T} = fill!(block.values, v)

blocktotal(block::Block) = size(block.values)
blockcells(block::Block{N, T, O}) where {N, T, O} = size(block.values) .- 2O
blockvalue(block::Block{N, T, O}, cell::CartesianIndex{N}) where {N, T, O} = block.values[CartesianIndex(cell.I .+ O)]
setblockvalue!(block::Block{N, T, O}, value::T, cell::CartesianIndex{N}) where {N, T, O} = block.values[CartesianIndex(cell.I .+ O)] = value

cellindices(block::Block) = CartesianIndices(blockcells(block))
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

export blockgradient, blockhessian

"""
Computes the gradient at a cell on a domain.
"""
@generated function blockgradient(block::Block{N, T, O}, cell::CartesianIndex{N}, basis::AbstractBasis{T}) where {N, T, O}
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
@generated function blockhessian(block::Block{N, T, O}, cell::CartesianIndex{N}, basis::AbstractBasis{T}) where {N, T, O}
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
    cells = blockcells(block)
    linear = LinearIndices(cells)

    for (i, cell) in enumerate(CartesianIndices(cells))
        setblockvalue!(block, f(i), cell)
    end
end

############################
## Boundary ################
############################

export BoundaryCell, block_surfaces, block_boundary, block_robin!, block_diritchlet!, block_nuemann!
export is_boundary_on_face

struct BoundaryCell{N, I}
    cell::CartesianIndex{N}

    BoundaryCell{I}(cell::CartesianIndex{N}) where {N, I} = new{N, I}(cell)
end

function is_boundary_on_face(::BoundaryCell{N, I}, face::FaceIndex) where {N, I}
    side = faceside(face)
    axis = faceaxis(face)

    if I[axis] == 1
        return side
    elseif I[axis] == -1
        return !side
    else
        return false
    end
end

"""
Iterates the subsurfaces of an `N`- dimensional block. The argument `Val(I)` is passed to `f`, where
`I` is an N-tuple of integers. 
"""
@generated function block_surfaces(f::F, ::Block{N, T, O}) where {N, T, O, F <: Function}
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
function block_boundary(f::F, block::Block{N, T, O}) where {N, T, O, F <: Function}
    block_surfaces(i -> _block_boundary(f, block, i), block)
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
            f(BoundaryCell{$I}(cell))
        end
    end
end

function _value_stencil_exprs(O, exterior::NTuple{N, Int}, I::NTuple{N, Int}) where N
    ntuple(Val(N)) do i
        if I[i] == 0
            return :(cell_value_stencil(basis, Val(0), Val(0)))
        elseif I[i] == 1
            return :(vertex_value_stencil(basis, Val($(2O + 1)), Val($(exterior[i])), Val(false)))
        else
            return :(vertex_value_stencil(basis, Val($(exterior[i])), Val($(2O + 1)), Val(true)))
        end
    end
end

function _gradient_stencil_exprs(O, exterior::NTuple{N, Int}, I::NTuple{N, Int}, axis::Int) where N
    ntuple(Val(N)) do i
        if i == axis
            if I[i] == 0
                return :(cell_derivative_stencil(basis, Val($O), Val($O)))
            elseif I[i] == 1
                return :(vertex_derivative_stencil(basis, Val($(2O)), Val($(exterior[i])), Val(false)))
            else
                return :(vertex_derivative_stencil(basis, Val($(exterior[i])), Val($(2O)), Val(true)))
            end
        else
            if I[i] == 0
                return :(cell_value_stencil(basis, Val(0), Val(0)))
            elseif I[i] == 1
                return :(vertex_value_stencil(basis, Val($(2O)), Val($(exterior[i])), Val(false)))
            else
                return :(vertex_value_stencil(basis, Val($(exterior[i])), Val($(2O)), Val(true)))
            end
        end
    end
end

function _stencil_edge_coefs_expr(I::NTuple{N, Int}, stencil::Symbol) where N
    coefs = Expr[]

    for i in 1:N
        if I[i] == -1
            push!(coefs, :($(stencil)[$i].left[end]))
        elseif I[i] == 1
            push!(coefs, :($(stencil)[$i].right[end]))
        end
    end

    :(*($(coefs...)))
end

@generated function block_robin!(block::Block{N, T, O}, boundary::BoundaryCell{N, I}, basis::AbstractBasis{T}, α::T, β::SVector{N, T}, c::T) where {N, T, O, I}
    exterior_cells = ntuple(i -> ifelse(I[i] ≠ 0, O, 1), Val(N))
    exterior_exprs = Expr[]

    for exterior in CartesianIndices(exterior_cells)
        cell_offset = exterior.I .* I

        # Value stencil
        value_stencil = _value_stencil_exprs(O, exterior.I, I)
        value_coefs = _stencil_edge_coefs_expr(I, :value_stencils)

        value_stencil_expr = quote
            value_stencils = tuple($(value_stencil...))
            result += value_scale * block_stencil_product(block, cell, value_stencils)
            coefs += value_scale * $value_coefs
        end

        gradient_stencil_exprs = Expr[]

        for axis in 1:N
            if I[axis] ≠ 0
                gradient_stencil = _gradient_stencil_exprs(O, exterior.I, I, axis)
                gradient_coefs = _stencil_edge_coefs_expr(I, :gradient_stencils)

                expr = quote
                    let 
                        gradient_stencils = tuple($(gradient_stencil...))
                        result += gradient_scale[$axis] * block_stencil_product(block, cell, gradient_stencils)
                        coefs += gradient_scale[$axis] * $gradient_coefs
                    end
                end

                push!(gradient_stencil_exprs, expr)
            end
        end

        # Final expresion for this exterior point
        expr = quote
            let 
                target = cell.I .+ $(cell_offset)

                setblockvalue!(block, zero($T), CartesianIndex(target))

                coefs = zero($T)
                result = zero($T)
                
                $(value_stencil_expr)
                $(gradient_stencil_exprs...)
                
                setblockvalue!(block, (homogenous - result)/coefs, CartesianIndex(target))
            end
        end

        push!(exterior_exprs, expr)
    end

    # Final result
    quote
        # Alias
        cell = boundary.cell
        
        value_scale = α
        gradient_scale = β .* blockcells(block) .* $I   
        homogenous = c

        $(exterior_exprs...)
    end
end

block_diritchlet!(block::Block{N, T}, boundary::BoundaryCell{N}, basis::AbstractBasis{T}, α::T, c::T) where {N, T} = block_robin!(block, boundary, basis, α, zero(SVector{N, T}), c)
block_nuemann!(block::Block{N, T}, boundary::BoundaryCell{N}, basis::AbstractBasis{T}, β::SVector{N, T}, c::T) where {N, T} = block_robin!(block, boundary, basis, zero(T), β, c)
