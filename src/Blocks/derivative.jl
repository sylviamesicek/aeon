#######################
## Derivatives ########
#######################

export block_value, block_gradient, block_hessian

"""
Computes the value of the block at a cell on the domain.
"""
block_value(block::AbstractBlock{N}, cell::CartesianIndex{N}) where N = blockvalue(block, cell)

"""
Computes the gradient at a cell on a domain.
"""
@generated function block_gradient(block::AbstractBlock{N, T, O}, cell::CartesianIndex{N}, basis::AbstractBasis{T}) where {N, T, O}
    quote 
        cells = blockcells(block)

        grad = Base.@ntuple $N i -> begin
            block_stencil_product(block, cell, gradient_stencils(basis, Val($N), Val($O), i)) * cells[i]
        end

        SVector{$N, $T}(grad) 
    end
end

"""
Computes the hessian at a cell on a domain.
"""
@generated function block_hessian(block::AbstractBlock{N, T, O}, cell::CartesianIndex{N}, basis::AbstractBasis{T}) where {N, T, O}
    quote
        cells = blockcells(block)

        hess = Base.@ntuple $(N*N) index -> begin
            i = (index - 1) ÷ $N + 1
            j = (index - 1) % $N + 1
            block_stencil_product(block, cell, hessian_stencils(basis, Val($N), Val($O), i, j)) * cells[i] * cells[j]
        end
        
        SMatrix{$N, $N, $T}(hess)
    end
end

#######################
## Diagonals ##########
#######################

export value_diagonal, gradient_diagonal, hessian_diagonal

function value_diagonal(::AbstractBasis{T}, ::Val{N}) where {N, T}
    one(T)
end

@generated function gradient_diagonal(basis::AbstractBasis{T}, ::Val{N}, ::Val{O}) where {N, T, O}
    quote 
        cells = blockcells(block)

        grad = Base.@ntuple $N i -> stencil_diagonal(gradient_stencils(basis,  Val($N), Val($O), i)) * cells[i]

        SVector{$N, $T}(grad) 
    end
end

@generated function hessian_diagonal(basis::AbstractBasis{T}, ::Val{N}, ::Val{O}) where {N, T, O}
    quote
        cells = blockcells(block)

        hess = Base.@ntuple $(N*N) index -> begin
            i = (index - 1) ÷ $N + 1
            j = (index - 1) % $N + 1
            stencil_diagonal(hessian_stencils(basis, Val($N), Val($O), i, j)) * cells[i] * cells[j]
        end
        
        SMatrix{$N, $N, $T}(hess)
    end
end