########################
## Abstract Block ######
########################

export AbstractBlock
export blocktotal, blockcells, blockvalue, setblockvalue!
export cellindices, cellwidths, cellposition

"""
Represents an abstract block of function data including a buffer region with width `O`.
"""
abstract type AbstractBlock{N, T, O} end

Base.size(block::AbstractBlock) = error("Unimplemented")
Base.fill!(block::AbstractBlock, v) = error("Unimplemented")
Base.getindex(block::AbstractBlock{N}, i::CartesianIndex{N}) where N = error("Unimplemented")
Base.setindex!(block::AbstractBlock{N, T}, v::T, i::CartesianIndex{N}) where {N, T} = error("Unimplemented")

blockbuffer(block::AbstractBlock) = error("Unimplemented")

blocktotal(block::AbstractBlock) = size(block)
blockcells(block::AbstractBlock{N, T, O}) where {N, T, O} = blocktotal(block) .- 2O
blockvalue(block::AbstractBlock{N, T, O}, cell::CartesianIndex{N}) where {N, T, O} = block[CartesianIndex(cell.I .+ O)]
setblockvalue!(block::AbstractBlock{N, T, O}, value::T, cell::CartesianIndex{N}) where {N, T, O} = block[CartesianIndex(cell.I .+ O)] = value

cellindices(block::AbstractBlock) = CartesianIndices(blockcells(block))
cellwidths(block::AbstractBlock{N, T}) where {N, T} = SVector{N, T}(1 ./ blockcells(block))
cellposition(block::AbstractBlock{N, T}, cell::CartesianIndex{N}) where {N, T} = SVector{N, T}((cell.I .- T(1//2)) ./ blockcells(block))

#########################
## Stencil Product ######
#########################

export block_stencil_product

"""
Apply the tensor product of a set of stencils at a cell on an `AbstractBlock`.
"""
function block_stencil_product(block::AbstractBlock{N, T}, cell::CartesianIndex{N}, stencils::NTuple{N, Stencil{T}}) where {N, T}
    _block_stencil_product(block, cell, stencils)
end

function _block_stencil_product(block::AbstractBlock{N, T}, cell::CartesianIndex{N}, ::NTuple{0, Stencil{T}}) where {N, T}
    blockvalue(block, cell)
end

@generated function _block_stencil_product(block::AbstractBlock{N, T}, cell::CartesianIndex{N}, stencils::S) where {N, T, M, S <: NTuple{M, Stencil{T}}}
    L, R = _stencil_supports(last(S.parameters))

    quote
        # Split tuple 
        stencil = stencils[$M]
        rest = Base.@ntuple $(M - 1) i -> stencils[i]

        result = stencil.center * _block_stencil_product(block, cell, rest)
        
        Base.@nexprs $L i -> begin
            result += stencil.left[i] * _block_stencil_product(block, CartesianIndex(setindex(cell.I, cell[$M] - i, $M)), rest)
        end

        Base.@nexprs $R i -> begin
            result += stencil.right[i] * _block_stencil_product(block, CartesianIndex(setindex(cell.I, cell[$M] + i, $M)), rest)
        end

        result
    end
end

#########################
## Iteration ############
#########################

export foreach_buffer, foreach_boundary

"""
Iterates the buffer regions of an `N` - dimensional block.
"""
@generated function foreach_buffer(f::F, ::AbstractBlock{N}) where {N, F <: Function}
    exprs = Expr[]

    for i in 1:N
        for subdomain in CartesianIndices(ntuple(_ -> 3, Val(N)))
            if sum(subdomain.I .== 1 .|| subdomain.I .== 3) == i
                push!(exprs, :(f(Val($(subdomain.I .- 2)))))
            end
        end
    end

    quote
        $(exprs...)
    end
end

"""
Iterates each boundary cell, along with associated buffer region.
"""
function foreach_boundary(f::F, block::AbstractBlock{N, T, O}) where {N, T, O, F <: Function}
    foreach_buffer(block) do i
        _foreach_boundary(f, block, i)
    end
end

@generated function _foreach_boundary(f::F, block::AbstractBlock{N, T, O}, ::Val{I}) where {N, T, O, I, F <: Function}
    cell_indices_expr = ntuple(i -> ifelse(I[i] == 0, :(1:cells[$i]), :(1:1)), Val(N))
    cell_expr = ntuple(Val(N)) do i
        if I[i] == 1
            :(cells[$i])
        elseif I[i] == -1
            :(1)
        else
            :(boundarycell[$i])
        end
    end

    quote
        cells = blockcells(block)

        for boundarycell in CartesianIndices(tuple($(cell_indices_expr...)))
            cell = CartesianIndex(tuple($(cell_expr...)))
            f(cell, Val($I))
        end
    end
end

########################
## Array Block #########
########################

export ArrayBlock

"""
A block backed by a multidimensional `Array`. This should provide the fastest (cpu) access to resources, 
and is prefered for complex computations.
"""
struct ArrayBlock{N, T, O} <: AbstractBlock{N, T, O}
    values::Array{T, N}

    ArrayBlock{O}(values::Array{T, N}) where {N, T, O} = new{N, T, O}(values)
end

ArrayBlock{T, O}(dims::Vararg{Int, N}) where {N, T, O} = ArrayBlock{O}(zeros(T, (dims .+ 2O)...))  
ArrayBlock{N, T, O}() where {N, T, O} = ArrayBlock{T, O}(ntuple(i -> 2O + 1, Val(N))...)

Base.size(block::ArrayBlock) = size(block.values)
Base.fill!(block::ArrayBlock{N, T}, v::T) where {N, T} = fill!(block.values, v)
Base.getindex(block::ArrayBlock{N}, i::CartesianIndex{N}) where N = block.values[i]
Base.setindex!(block::ArrayBlock{N, T}, v::T, i::CartesianIndex{N}) where {N, T} = block.values[i] = v

########################
## View Block ##########
########################

export ViewBlock

"""
A block basked by a linear array.
"""
struct ViewBlock{N, T, A} <: AbstractBlock{N, T, 0} 
    values::A
    offset::Int
    dims::NTuple{N, Int}

    ViewBlock(values::AbstractVector{T}, offset::Int, dims::NTuple{N, Int}) where {N, T} = new{N, T, typeof(values)}(values, offset, dims)
end

Base.size(block::ViewBlock) = block.dims

function Base.fill!(block::ViewBlock{N, T}, v::T) where {N, T}
    total = prod(block.dims)
    view = @view block.values[(1:total) .+ block.offset]
    fill!(view, v)
end    

function Base.getindex(block::ViewBlock{N}, i::CartesianIndex{N}) where N
    linear = LinearIndices(block.dims)
    block.values[bloc.offset + linear[i]]
end

function Base.setindex!(block::ViewBlock{N, T}, v::T, i::CartesianIndex{N}) where {N, T}
    linear = LinearIndices(block.dims)
    block.values[bloc.offset + linear[i]] = v
end

############################
## Helpers #################
############################

"""
Helper function for extracting the support widths of a stencil type.
"""
_stencil_supports(::Type{Stencil{T, L, R}}) where {T, L, R} = (L, R)
