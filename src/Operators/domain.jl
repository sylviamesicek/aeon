
#########################
## Domain ###############
#########################

export Domain, domaincells, domainvalue

"""
A block with fictious nodes filled.
"""
struct Domain{N, T, O}
    inner::Array{T, N}
end

Domain{N, T, O}(::UndefInitializer) where {N, T, O} = Domain{N, T, O}(Array{T, N}(undef, ntuple(i -> 0, Val(N)))) 
Domain{O}(::UndefInitializer, block::Block{N, T}) where {N, T, O} = Domain{N, T, O}(Array{T, N}(undef, (blockcells(block) .+ 2O)...))

domaincells(domain::Domain{N, T, O}) where {N, T, O} = size(domain.inner) .- 2O
domainvalue(domain::Domain{N, T, O}, cell::CartesianIndex{N}) where {N, T, O} = domain.inner[CartesianIndex(cell.I .+ O)]

Base.fill!(domain::Domain{N, T}, v::T) where {N, T}= fill!(domain.inner, v)

# Helper
setdomainvalue!(domain::Domain{N, T, O}, value::T, cell::CartesianIndex{N}) where {N, T, O} = domain.inner[CartesianIndex(cell.I .+ O)] = value

########################
## Stencils ############
########################

export domain_stencil_product

"""
Apply the tensor product of a set of stencils at a point on a numerical domain
"""
function domain_stencil_product(domain::Domain{N, T}, cell::CartesianIndex{N}, stencils::NTuple{N, Stencil{T}}) where {N, T}
    _domain_stencil_product(domain, cell, stencils)
end

function _domain_stencil_product(domain::Domain{N, T}, cell::CartesianIndex{N}, ::NTuple{0, Stencil{T}}) where {N, T}
    domainvalue(domain, cell)
end

function _domain_stencil_product(domain::Domain{N, T}, cell::CartesianIndex{N}, stencils::NTuple{L, Stencil{T}}) where {N, T, L}
    remaining = ntuple(i -> stencils[i], Val(L - 1))

    result = stencils[L].center * _domain_stencil_product(domain, cell, remaining)

    for (i, left) in enumerate(stencils[L].left)
        offcell = CartesianIndex(setindex(cell, cell[L] - i, L))
        result += left * _domain_stencil_product(domain, offcell, remaining)
    end

    for (i, right) in enumerate(stencils[L].right)
        offcell = CartesianIndex(setindex(cell, cell[L] + i, L))
        result += right * _domain_stencil_product(domain, offcell, remaining)
    end

    result
end

##############################
## Evaluation ################
##############################

export domainevaluate, domaingradient, domainhessian

"""
Evaluates the tensor product of a set of abstract operators at a point on a domain.
"""
function domainevaluate(domain::Domain{N, T, O}, cell::CartesianIndex{N}, basis::AbstractBasis{T}, opers::NTuple{N, AbstractOperator}) where {N, T, O}
    stencils = map(i -> operator_stencil(basis, Val(O), i), opers)
    domain_stencil_product(domain, cell, stencils)
end


"""
Computes the gradient at a cell on a domain.
"""
@generated function domaingradient(domain::Domain{N, T, O}, cell::CartesianIndex{N}, basis::AbstractBasis{T}) where {N, T, O}
    quote 
        cells = size(domain.inner) .- 2O

        grad = Base.@ntuple $N i -> begin
            opers = Base.@ntuple $N dim -> ifelse(i == dim, ValueOperator{1}(), ValueOperator{0}())
            domainevaluate(domain, cell, basis, opers) * cells[i]
        end

        SVector(grad) 
    end
end

"""
Computes the hessian at a cell on a domain.
"""
@generated function domainhessian(domain::Domain{N, T, O}, cell::CartesianIndex{N}, basis::AbstractBasis{T}) where {N, T, O}
    quote
        cells = size(domain.inner) .- $(2O)

        hess = Base.@ntuple $(N*N) index -> begin
            i = (index - 1) ÷ $N + 1
            j = (index - 1) % $N + 1
            if i == j
                opers = Base.@ntuple $N dim -> ifelse(i == dim, ValueOperator{2}(), ValueOperator{0}())
                result = domainevaluate(domain, cell, basis, opers) * cells[i]^2
            else
                opers = Base.@ntuple $N dim -> ifelse(i == dim || j == dim, ValueOperator{1}(), ValueOperator{0}())
                result = domainevaluate(domain, cell, basis, opers) * cells[i] * cells[j]
            end

            result
        end
        
        SMatrix{$N, $N, $T}(hess)
    end
end

#############################
## Prolongation #############
#############################

export domainprolong

"""
Performs prolongation for a full domain.
"""
function domainprolong(domain::Domain{N, T, O}, point::NTuple{N, PointIndex}, basis::AbstractBasis{T}) where {N, T, O}
    cell = CartesianIndex(map(point_to_cell, point))
    stencils = map(i -> _point_to_prolong_stencil(i, basis, Val(O)), point) 
    domain_stencil_product(domain, cell, stencils)
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

#########################
## Transfer #############
#########################

export transfer_block_to_domain!, interface_value, interface_gradient, interface_homogenous

interface_value(::Field{N, T}, ::Block{N, T}, ::CartesianIndex{N}, ::AbstractBasis{T}, ::Val{O}, ::Val{I}) where {N, T, O, I} = error("Unimplemented")
interface_gradient(::Field{N, T}, ::Block{N, T}, ::CartesianIndex{N}, ::AbstractBasis{T}, ::Val{O}, ::Val{I}) where {N, T, O, I} = error("Unimplemented")
interface_homogenous(::Field{N, T}, ::Block{N, T}, ::CartesianIndex{N}, ::AbstractBasis{T}, ::Val{O}, ::Val{I}) where {N, T, O, I} = error("Unimplemented")

"""
Fills a subdomain of the boundary of a domain.
"""
@generated function fill_interface!(domain::Domain{N, T, O}, field::Field{N, T}, block::Block{N, T}, cell::CartesianIndex{N}, basis::AbstractBasis{T}, ::Val{I}) where {N, T, O, I}
    exterior_cells = ntuple(i -> ifelse(I[i] ≠ 0, O, 1), Val(N))
    exterior_exprs = Expr[]

    for exterior in CartesianIndices(exterior_cells)
        cell_offset = exterior.I .* I

        # Value stencil
        value_stencil = ntuple(Val(N)) do i
            if I[i] == 0
                return :(cell_value_stencil(basis, Val(0), Val(0)))
            elseif I[i] == 1
                return :(vertex_value_stencil(basis, Val($(2O + 1)), Val($(exterior[i])), Val(false)))
            else
                return :(vertex_value_stencil(basis, Val($(exterior[i])), Val($(2O + 1)), Val(true)))
            end
        end

        value_coefs_exprs = Expr[]

        for i in 1:N
            if I[i] == -1
                push!(value_coefs_exprs, :(value_stencils[$i].left[end]))
            elseif I[i] == 1
                push!(value_coefs_exprs, :(value_stencils[$i].right[end]))
            end
        end

        value_stencil_expr = quote
            value_stencils = tuple($(value_stencil...))
            result += boundary_value * domain_stencil_product(domain, cell, value_stencils)
            coefs += *(boundary_value, $(value_coefs_exprs...))
        end

        # Gradient stencils
        gradient_stencil = axis -> begin
            ntuple(Val(N)) do i
                if i == axis
                    if I[i] == 0
                        return :(cell_derivative_stencil(basis, Val(O), Val(O)))
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

        gradient_stencil_exprs = Expr[]

        for axis in 1:N
            if I[axis] ≠ 0
                derivative_coefs_exprs = Expr[]

                for i in 1:N
                    if I[i] == -1
                        push!(derivative_coefs_exprs, :(derivative_stencils[$i].left[end]))
                    elseif I[i] == 1
                        push!(derivative_coefs_exprs, :(derivative_stencils[$i].right[end]))
                    end
                end

                expr = quote
                    let 
                        derivative_stencils = tuple($(gradient_stencil(axis)...))
                        result += boundary_gradient[$axis] * $(I[axis]) * domain_stencil_product(domain, cell, derivative_stencils)
                        coefs += *(boundary_gradient[$axis], $(I[axis]), $(derivative_coefs_exprs...))
                    end
                end

                push!(gradient_stencil_exprs, expr)
            end
        end

        # Final expresion for this exterior point
        expr = quote
            let 
                coefs = zero(T)
                result = zero(T)

                $(value_stencil_expr)
                $(gradient_stencil_exprs...)

                target = cell.I .+ $(cell_offset)
                
                setdomainvalue!(domain, (homogenous - result)/coefs, CartesianIndex(target))
            end
        end

        push!(exterior_exprs, expr)
    end

    # Final result
    quote
        boundary_value = interface_value(field, block, cell, basis, Val($O), Val($I))
        boundary_gradient = interface_gradient(field, block, cell, basis, Val($O), Val($I))
        homogenous = interface_homogenous(field, block, cell, basis, Val($O), Val($I))
        # boundary_value = one($T)
        # boundary_gradient = zero(SVector{$N, $T})
        # homogenous = zero($T)

        $(exterior_exprs...)
    end
end

"""
Transfers the data of a field on a block to a domain. This is essentally a preprocessing step which allows
all subsequent operations on the domain to be much cheaper.
"""
function transfer_block_to_domain!(domain::Domain{N, T, O}, field::Field{N, T}, block::Block{N, T}, basis::AbstractBasis{T}) where {N, T, O}
    @assert domaincells(domain) == blockcells(block)

    fill!(domain, zero(T))

    # Fill interior values 
    for cell in cellindices(block)
        v = value(field, block, cell)
        domain.inner[CartesianIndex(cell.I .+ O)] = v
    end

    # Fill exterior values
    _fill_interfaces!(domain, field, block, basis)
end

@generated function _fill_interfaces!(domain::Domain{N, T, O}, field::Field{N, T}, block::Block{N, T}, basis::AbstractBasis{T}) where {N, T, O}
    # For a given i, find all subdomains with that number of edges.
    subdomain_exprs = i -> begin
        sub_exprs = Expr[]
        
        for subdomain in CartesianIndices(ntuple(_ -> 3, Val(N)))
            if sum(subdomain.I .== 1 .|| subdomain.I .== 3) == i
                push!(sub_exprs, :(_fill_interface!(domain, field, block, basis, Val($(subdomain.I .- 2)))))
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

@generated function _fill_interface!(domain::Domain{N, T, O}, field::Field{N, T}, block::Block{N, T}, basis::AbstractBasis{T}, ::Val{I}) where {N, T, O, I}
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
            fill_interface!(domain, field, block, cell, basis, Val($I))
        end
    end
end