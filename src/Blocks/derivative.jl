
##############################
## Diagonals #################
##############################

export block_value_diagonal, block_gradient_diagonal, block_hessian_diagonal

function block_value_diagonal(::AbstractBlock{N, T}, ::AbstractBasis{T}) where {N, T}
    one(T)
end

@generated function block_gradient_diagonal(block::AbstractBlock{N, T, O}, basis::AbstractBasis{T}) where {N, T, O}
    quote 
        cells = blockcells(block)

        grad = Base.@ntuple $N i -> stencil_diagonal(gradient_stencils(block, basis, i)) * cells[i]

        SVector{$N, $T}(grad) 
    end
end

@generated function block_hessian_diagonal(block::AbstractBlock{N, T, O}, basis::AbstractBasis{T}) where {N, T, O}
    quote
        cells = blockcells(block)

        hess = Base.@ntuple $(N*N) index -> begin
            i = (index - 1) ÷ $N + 1
            j = (index - 1) % $N + 1
            stencil_diagonal(hessian_stencils(block, basis, i, j)) * cells[i] * cells[j]
        end
        
        SMatrix{$N, $N, $T}(hess)
    end
end

#############################
## Derivatives ##############
#############################

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
            block_stencil_product(block, cell, gradient_stencils(block, basis, i)) * cells[i]
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
            block_stencil_product(block, cell, hessian_stencils(block, basis, i, j)) * cells[i] * cells[j]
        end
        
        SMatrix{$N, $N, $T}(hess)
    end
end

##########################
## Stencils ##############
##########################

@generated function value_stencils(::AbstractBlock{N, T, O}, basis::AbstractBasis{T}) where {N, T, O}
    :(Base.@ntuple $N dim -> Stencil(basis, CovariantDerivative{O, 0}()))
end

@generated function gradient_stencils(::AbstractBlock{N, T, O}, basis::AbstractBasis{T}, i) where {N, T, O}
    :(Base.@ntuple $N dim -> ifelse(i == dim, Stencil(basis, CovariantDerivative{O, 1}()), Stencil(basis, CovariantDerivative{O, 0}())))
end

@generated function hessian_stencils(::AbstractBlock{N, T, O}, basis::AbstractBasis{T}, i, j) where {N, T, O}
    quote
        if i == j
            Base.@ntuple $N dim -> ifelse(i == dim, Stencil(basis, CovariantDerivative{O, 2}()), Stencil(basis, CovariantDerivative{O, 0}()))
        else
            Base.@ntuple $N dim -> ifelse(i == dim || j == dim, Stencil(basis, CovariantDerivative{O, 1}()), Stencil(basis, CovariantDerivative{O, 0}()))
        end
    end
end