# Helpers for filling the interior and buffer region of blocks.

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
## Boundary ############
########################

export fill_boundaries!

function fill_boundaries!(f::F, block::AbstractBlock{N, T}, basis::AbstractBasis{T}) where {N, T, F <: Function}    
    foreach_boundary(block) do cell, i
        _fill_boundary!(f, block, basis, cell, i)
    end
end

@generated function _fill_boundary!(f::F, block::AbstractBlock{N, T, O}, basis::AbstractBasis{T}, cell::CartesianIndex{N}, ::Val{I}) where {N, T, O, I, F <: Function}
    exprs = Expr[]
    
    exterior_cells = ntuple(i -> ifelse(I[i] ≠ 0, O, 1), N)
    
    for exterior in CartesianIndices(exterior_cells)
        cell_offset = exterior.I .* I

        value_stencil = ntuple(N) do i
            if I[i] == 0
                return :(Stencil(basis, CellValue{$O, $O}()))
            elseif I[i] == 1
                return :(Stencil(basis, VertexValue{$(2O + 1), $(exterior[i]), false}()))
            else
                return :(Stencil(basis, VertexValue{$(exterior[i]), $(2O + 1), true}()))
            end
        end

        value_coefs = ntuple(N) do i
            if I[i] == 0
                return one(T)
            elseif I[i] == 1
                return :(stencils[$i].right[end])
            else
                return :(stencils[$i].left[end])
            end
        end

        push!(exprs, quote
            let 
                target = cell.I .+ $(cell_offset)
                # Avoid having to fill block with 0.0 beforehand
                setblockvalue!(block, zero($T), CartesianIndex(target))

                value = values[$exterior]
                stencils = tuple($(value_stencil...))
                result = block_stencil_product(block, cell, stencils)
                coef = prod(tuple($(value_coefs...)))

                setblockvalue!(block, (value - result) / coef, CartesianIndex(target))
            end
        end)
    end

    quote
        values = f(cell, Val(I))
        $(exprs...)
    end
end

########################
## Physical Boundary ###
########################

export BoundaryCondition, diritchlet, nuemann, robin

struct BoundaryCondition{T}
    value::T
    normal::T
    rhs::T
end

"""
Inhomogenous diritchlet boundary condition.
"""
diritchlet(value::T, rhs::T) where T = BoundaryCondition(value, zero(T), rhs)

"""
Inhomogenous nuemann boundary condition.
"""
nuemann(normal::T, rhs::T) where T = BoundaryCondition(zero(T), normal, rhs)

"""
Inhomogenous robin boundary condition.
"""
robin(value::T, normal::T, rhs::T) where T = BoundaryCondition(value, normal, rhs)