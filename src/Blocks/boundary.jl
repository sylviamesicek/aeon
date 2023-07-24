############################
## Boundary Cell ###########
############################

export BoundaryCell, boundary_edges, boundary_edge

struct BoundaryCell{N, I}
    cell::CartesianIndex{N}

    BoundaryCell{I}(cell::CartesianIndex{N}) where {N, I} = new{N, I}(cell)
end

boundary_edges(::BoundaryCell{N, I}) where {N, I} = I
boundary_edge(::BoundaryCell{N, I}, axis::Int) where {N, I} = I[axis]

############################
## Boundary ################
############################

export block_surfaces, block_boundary
"""
Iterates the subsurfaces of an `N`- dimensional block. The argument `Val(I)` is passed to `f`, where
`I` is an N-tuple of integers. 
"""
@generated function block_surfaces(f::F, ::AbstractBlock{N, T, O}) where {N, T, O, F <: Function}
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
function block_boundary(f::F, block::AbstractBlock{N, T, O}) where {N, T, O, F <: Function}
    block_surfaces(i -> _block_boundary(f, block, i), block)
end

@generated function _block_boundary(f::F, block::AbstractBlock{N, T, O}, ::Val{I}) where {N, T, O, I, F <: Function}
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

#################################
## Specific Boundary Conditions #
#################################

export block_boundary_condition!

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

@generated function block_boundary_condition!(block::AbstractBlock{N, T, O}, boundary::BoundaryCell{N, I}, basis::AbstractBasis{T}, α::T, β::SVector{N, T}, c::T) where {N, T, O, I}
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

##############################
## Public API ################
##############################

export BoundaryCondition, diritchlet, nuemann, robin
export block_boundary_conditions!

"""
Represents an arbitray robin boundary condition, including a value coefficient, a normal coefficient, and a right-hand-side.
"""
struct BoundaryCondition{T}
    value::T
    normal::T
    rhs::T
end

"""
Homogenous diritchlet boundary condition.
"""
diritchlet(value::T) where T = diritchlet(value, zero(T))

"""
Inhomogenous diritchlet boundary condition.
"""
diritchlet(value::T, rhs::T) where T = BoundaryCondition(value, zero(T), rhs)

"""
Homogenous nuemann boundary condition.
"""
nuemann(normal::T) where T = nuemann(normal, zero(T))

"""
Inhomogenous nuemann boundary condition.
"""
nuemann(normal::T, rhs::T) where T = BoundaryCondition(zero(T), normal, rhs)

"""
Homogenous robin boundary condition.
"""
robin(value::T, normal::T) where T = robin(value, normal, zero(T))

"""
Inhomogenous robin boundary condition.
"""
robin(value::T, normal::T, rhs::T) where T = BoundaryCondition(value, normal, rhs)

"""
Accumulates boundary condition using an identity transform for the normal vector
"""
block_boundary_conditions!(f::F, block::AbstractBlock{N, T}, basis::AbstractBasis{T}) where {N, T, F<: Function} = block_boundary_conditions!(f, block, basis, IdentityTransform{N, T}())

"""
Accumulates boundary conditions for each face of a block, and fills the corresponding ghost cells.
"""
function block_boundary_conditions!(f::F, block::AbstractBlock{N, T, O}, basis::AbstractBasis{T}, block_to_global::Transform{N, T}) where {N, T, O, F <: Function}
    block_boundary(block) do boundary
        lpos = cellposition(block, boundary.cell)
        j = jacobian(block_to_global, lpos)

        values = ntuple(Val(N)) do axis
            if boundary_edge(boundary, axis) ≠ 0
                f(boundary, axis)
            else
                BoundaryCondition{T}(0, 0, 0)
            end
        end

        α = sum(map(v -> v.value, values))
        β = j * SVector(map(v -> v.normal, values))
        c = sum(map(v -> v.rhs, values))

        block_boundary_condition!(block, boundary, basis, α, β, c)
    end
end
