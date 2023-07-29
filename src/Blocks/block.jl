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

"""
Returns the total number of cells in each dimension including ghost cells.
"""
blocktotal(block::AbstractBlock) = error("Blocktotal is unimplemented for $(typeof(block)).")

"""
Returns the number of cells in each dimension excluding ghost cells.
"""
blockcells(block::AbstractBlock) = error("Blockcells is unimplemented for $(typeof(block)).")

"""
Returns the value of a block at the given cell.
"""
blockvalue(block::AbstractBlock{N}, ::CartesianIndex{N}) where N = error("Blockvalue is unimplemented for $(typeof(block)).")

"""
Sets the value of a block at the given cell
"""
setblockvalue!(block::AbstractBlock{N, T}, ::T, ::CartesianIndex{N}) where {N, T} =  error("setblockvalue! is unimplemented for $(typeof(block)).")


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
    _block_stencil_product(block, cell.I, reverse(stencils)...)
end

function _block_stencil_product(block::AbstractBlock{N, T}, cell::NTuple{N, Int}) where {N, T}
    blockvalue(block, CartesianIndex(cell))
end

@generated function _block_stencil_product(block::AbstractBlock{N, T}, cell::NTuple{N, Int}, stencil::Stencil{T, L, R}, rest::Vararg{Stencil{T}, M}) where {N, T, M, L, R}
    axis = M + 1
    
    quote
        result = stencil.center .* _block_stencil_product(block, cell, rest...)

        Base.@nexprs $L i -> begin
            result = result .+ stencil.left[i] .* _block_stencil_product(block, setindex(cell, cell[$axis] - i, $axis), rest...)
        end

        Base.@nexprs $R i -> begin
            result = result .+ stencil.right[i] .* _block_stencil_product(block, setindex(cell, cell[$axis] + i, $axis), rest...)
        end

        result
    end
end


##########################
## Interior ##############
##########################

export fill_interior!, fill_interior_from_linear!

"""
Fills the interior of a block by calling `f` once for each interior cell (where the argument is a cartesian cell index).
"""
function fill_interior!(f::F, block::AbstractBlock) where {F <: Function}
    for cell in cellindices(block)
        setblockvalue!(block, f(cell), cell)
    end
end

"""
Fills the interior of a block by calling `f` once for each interior cell (where the argument is a linear cell index).
"""
function fill_interior_from_linear!(f::F, block::AbstractBlock) where {F <: Function}
    for (i, cell) in enumerate(cellindices(block))
        setblockvalue!(block, f(i), cell)
    end
end


########################
## View Block ##########
########################

# export ViewBlock

# """
# A block basked by a linear array.
# """
# struct ViewBlock{N, T, A} <: AbstractBlock{N, T, 0} 
#     values::A
#     offset::Int
#     dims::NTuple{N, Int}

#     ViewBlock(values::AbstractVector{T}, offset::Int, dims::NTuple{N, Int}) where {N, T} = new{N, T, typeof(values)}(values, offset, dims)
# end

# Base.size(block::ViewBlock) = block.dims

# function Base.fill!(block::ViewBlock{N, T}, v::T) where {N, T}
#     total = prod(block.dims)
#     view = @view block.values[(1:total) .+ block.offset]
#     fill!(view, v)
# end    

# function Base.getindex(block::ViewBlock{N}, i::CartesianIndex{N}) where N
#     linear = LinearIndices(block.dims)
#     block.values[bloc.offset + linear[i]]
# end

# function Base.setindex!(block::ViewBlock{N, T}, v::T, i::CartesianIndex{N}) where {N, T}
#     linear = LinearIndices(block.dims)
#     block.values[bloc.offset + linear[i]] = v
# end
