#########################
## Array Block ##########
#########################

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

blocktotal(block::ArrayBlock) = size(block.values)
blockcells(block::ArrayBlock{N, T, O}) where {N, T, O} = size(block.values) .- 2O
blockvalue(block::ArrayBlock{N, T, O}, cell::CartesianIndex{N}) where {N, T, O} = block.values[(cell.I .+ O)...]
setblockvalue!(block::ArrayBlock{N, T, O}, v::T, cell::CartesianIndex{N}) where {N, T, O} =  block.values[(cell.I .+ O)...] = v

#########################
## Iteration ############
#########################

export foreach_buffer, foreach_boundary

"""
Iterates the buffer regions of an `N` - dimensional block.
"""
@generated function foreach_buffer(f::F, ::ArrayBlock{N}) where {N, F <: Function}
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
function foreach_boundary(f::F, block::ArrayBlock{N, T, O}) where {N, T, O, F <: Function}
    foreach_buffer(block) do i
        _foreach_boundary(f, block, i)
    end
end

@generated function _foreach_boundary(f::F, block::ArrayBlock{N, T, O}, ::Val{I}) where {N, T, O, I, F <: Function}
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
## Boundary ############
########################

export fill_boundaries!

function fill_boundaries!(f::F, block::ArrayBlock{N, T}, basis::AbstractBasis{T}) where {N, T, F <: Function}    
    foreach_boundary(block) do cell, i
        _fill_boundary!(f, block, basis, cell, i)
    end
end

@generated function _fill_boundary!(f::F, block::ArrayBlock{N, T, O}, basis::AbstractBasis{T}, cell::CartesianIndex{N}, ::Val{I}) where {N, T, O, I, F <: Function}
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