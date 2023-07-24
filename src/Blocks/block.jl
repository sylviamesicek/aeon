########################
## Abstract Block ######
########################

export AbstractBlock
export blocktotal, blockcells, blockvalue, setblockvalue!
export cellindices, cellwidths, cellposition

abstract type AbstractBlock{N, T, O} end

Base.size(block::AbstractBlock) = error("Unimplemented")
Base.fill!(block::AbstractBlock, v) = error("Unimplemented")
Base.getindex(block::AbstractBlock{N}, i::CartesianIndex{N}) where N = error("Unimplemented")
Base.setindex!(block::AbstractBlock{N, T}, v::T, i::CartesianIndex{N}) where {N, T} = error("Unimplemented")

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
Apply the tensor product of a set of stencils at a point on a numerical domain
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
            result += stencil.left[i] * _block_stencil_product(block, CartesianIndex(setindex(cell, cell[$M] - i, $M)), rest)
        end

        Base.@nexprs $R i -> begin
            result += stencil.right[i] * _block_stencil_product(block, CartesianIndex(setindex(cell, cell[$M] + i, $M)), rest)
        end

        result
    end
end

#######################
## Derivatives ########
#######################

export blockgradient, blockhessian

"""
Computes the gradient at a cell on a domain.
"""
@generated function blockgradient(block::AbstractBlock{N, T, O}, cell::CartesianIndex{N}, basis::AbstractBasis{T}) where {N, T, O}
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
@generated function blockhessian(block::AbstractBlock{N, T, O}, cell::CartesianIndex{N}, basis::AbstractBasis{T}) where {N, T, O}
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

#############################
## Transfer #################
#############################

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
## Array Block #########
########################

export ArrayBlock

struct ArrayBlock{N, T, O} <: AbstractBlock{N, T, O}
    values::Array{T, N}

    ArrayBlock{O}(values::Array{T, N}) where {N, T, O} = new{N, T, O}(values)
end

ArrayBlock{T, O}(::UndefInitializer, dims::Vararg{Int, N}) where {N, T, O} = ArrayBlock{O}(Array{T, N}(undef, (dims .+ 2O)...))  
ArrayBlock{N, T, O}(::UndefInitializer) where {N, T, O} = ArrayBlock{T, O}(undef, ntuple(i -> 2O + 1, Val(N))...)

Base.size(block::ArrayBlock) = size(block.values)
Base.fill!(block::ArrayBlock{N, T}, v::T) where {N, T} = fill!(block.values, v)
Base.getindex(block::ArrayBlock{N}, i::CartesianIndex{N}) where N = block.values[i]
Base.setindex!(block::ArrayBlock{N, T}, v::T, i::CartesianIndex{N}) where {N, T} = block.values[i] = v

########################
## View Block ##########
########################

export ViewBlock

struct ViewBlock{N, T, O, A} <: AbstractBlock{N, T, O}
    values::A

    ViewBlock{O}(values::AbstractArray{T, N}) where {N, T, O} = new{N, T, O, typeof(values)}(values)
end

Base.size(block::ViewBlock) = size(block.values)
Base.fill!(block::ViewBlock{N, T}, v::T) where {N, T} = fill!(block.values, v)
Base.getindex(block::ViewBlock{N}, i::CartesianIndex{N}) where N = block.values[i]
Base.setindex!(block::ViewBlock{N, T}, v::T, i::CartesianIndex{N}) where {N, T} = block.values[i] = v

############################
## Helpers #################
############################

_stencil_supports(::Type{Stencil{T, L, R}}) where {T, L, R} = (L, R)
