#####################
## Points ###########
#####################

export PointIndex, CellIndex, SubCellIndex, VertexIndex
export subcell_side, subcell_to_cell, vertex_to_cell

"""
An abstract point index on a structured block grid.
"""
abstract type PointIndex end

"""
The index of a cell.
"""
struct CellIndex <: PointIndex
    inner::Int
end

point_to_cell(v::CellIndex) = c.inner

"""
A subcell index (ie, the two subcells along each axis which subdivide a cell)
"""
struct SubCellIndex <: PointIndex
    inner::Int
end

"""
Computes the side of this subcells subdivision
"""
subcell_side(v::SubCellIndex) = v.inner % 2 == 1

point_to_cell(v::SubCellIndex) = CellIndex((v.inner + 1) ÷ 2)

"""
A vertex index.
"""
struct VertexIndex <: PointIndex
    inner::Int
end

point_to_cell(v::VertexIndex) = CellIndex(v.inner - 1)

########################
## Block ###############
########################

export Block, blockcells, blockbounds, blocktransform
export cellindices, cellwidths, cellcenter
export Field, value, setvalue!
export blockprolong

abstract type Block{N, T} end

"""
Returns the number of cells that make up this block (must be implemented for each block type).
"""
blockcells(::Block) = error("Unimplemented")

"""
Returns the bounds of this block (must be implemented for each block type).
"""
blockbounds(::Block) = error("Unimplemented") 

"""
Returns a transform from block space to global space.
"""
function blocktransform(block::Block)
    bounds = blockbounds(block)
    Translate(bounds.origin) ∘ ScaleTransform(bounds.widths)
end

"""
Iterates the indices of a given block.
"""
cellindices(block::Block) = CartesianIndices(domaincells(block))
cellwidths(block::Block{N, T}) where {N, T} = SVector{N, T}(1 ./ blockcells(block))
cellcenter(block::Block{N, T}, cell::CartesianIndex{N}) where {N, T} = SVector{N, T}((cell.I .- T(1//2)) ./ blockcells(block))

# Field

"""
An abstract field defined over a domain.
"""
abstract type Field{N, T} end

"""
Retrieves or computes the value of a field at a cell in a block (must be implemented for each field type).
"""
value(field::Field{N, T}, block::Block{N, T}, cell::CartesianIndex{N}) where {N, T} = error("Unimplemented")

"""
Sets the value of a field at a cell in a block (must be implemented for each field type).
"""
setvalue!(field::Field{N, T}, value::T, block::Block{N, T}, cell::CartesianIndex{N}) where {N, T} = error("Unimplemented")

#############################
## Prolongation (Block) #####
#############################

"""
Apply the tensor product of a set of stencils at a point on a block.
"""
function block_stencil_product(field::Field{N, T}, block::Block{N, T}, cell::CartesianIndex{N}, stencils::NTuple{N, AbstractStencil{T}}) where {N, T}
    _block_stencil_product(domain, cell, stencils...)
end

function _block_stencil_product(field::Field{N, T}, block::Block{N, T}, cell::CartesianIndex{N}) where {N, T, L}
    value(field, block, cell)
end

function _block_stencil_product(field::Field{N, T}, block::Block{N, T}, cell::CartesianIndex{N}, stencil::CellStencil{T}, rest::Vararg{AbstractStencil{T}, L}) where {N, T, L}
    axis = N - L
    result = stencil.center * _block_stencil_product(field, block, cell, rest...)

    for (i, left) in enumerate(stencil.left)
        offcell = CartesianIndex(setindex(cell, cell[axis] - i, axis))
        result += left * _block_stencil_product(field, block, offcell, rest...)
    end

    for (i, right) in enumerate(stencil.right)
        offcell = CartesianIndex(setindex(cell, cell[axis] + i, axis))
        result += right * _block_stencil_product(field, block, offcell, rest...)
    end

    result
end

function _block_stencil_product(field::Field{N, T}, block::Block{N, T}, cell::CartesianIndex{N}, stencil::VertexStencil{T}, rest::Vararg{AbstractStencil{T}, L}) where {N, T, L}
    axis = N - L
    result = zero(T)

    for (i, left) in enumerate(stencil.left)
        offcell = CartesianIndex(setindex(cell, cell[axis] + 1 - i, axis))
        result += left * _domain_stencil_product(field, block, offcell, rest...)
    end

    for (i, right) in enumerate(stencil.right)
        offcell = CartesianIndex(setindex(cell, cell[axis] + i, axis))
        result += right * _domain_stencil_product(field, block, offcell, rest...)
    end

    result
end

"""
Performs prolongation within a block, to the given order.
"""
function blockprolong(field::Field{N, T}, block::Block{N, T}, point::NTuple{N, PointIndex}, basis::AbstractBasis{T}, ::Val{O}) where {N, T, O}
    _blockprolong(field, block, basis, (), (), point...)
end

function _blockprolong(field::Field{N, T}, block::Block{N, T}, ::AbstractBasis{T}, ::Val{O}, cell::NTuple{N, Int}, stencils::NTuple{N, AbstractStencil}) where {N, T, O}
    block_stencil_product(field, block, CartesianIndex(cell), stencils)
end

function _blockprolong(field::Field{N, T}, block::Block{N, T}, basis::AbstractBasis{T}, ::Val{O}, cell::NTuple{L, Int}, stencils::NTuple{L, AbstractStencil}, index::CellIndex, rest::Vararg{PointIndex, M}) where {N, T, O, L, M}
    stencil = cell_value_stencil(basis, Val(O), Val(O))
    _blockprolong(field, block, basis, Val(O), (cell..., index), (stencils..., stencil), rest...)
end

@generated function _blockprolong(field::Field{N, T}, block::Block{N, T}, basis::AbstractBasis{T}, ::Val{O}, cell::NTuple{L, Int}, stencils::NTuple{L, AbstractStencil}, index::VertexIndex, rest::Vararg{PointIndex, M}) where {N, T, O, L, M}
    quote
        ctotal = blockcells(block)[$L]
        cindex = vertex_to_cell(index)

        leftcells = min($O, cindex)
        rightcells = min($O, total - cindex)

        # Left side
        if leftcells < $O
            Base.@nexprs $O i -> begin
                if leftcells == i - 1
                    return _blockprolong(field, block, basis, Val(O), 
                        (cell..., cindex), 
                        (stencils..., vertex_value_stencil(basis, Val(i - 1), Val(2* $O + 1))), 
                    rest...)
                end
            end
        end

        # Right side
        if rightcells < $O
            Base.@nexprs $O i -> begin
                if rightcells == i - 1
                    return _blockprolong(field, block, basis, Val(O), 
                        (cell..., cindex), 
                        (stencils..., vertex_value_stencil(basis, Val(2* $O + 1), Val(i - 1))), 
                    rest...)
                end
            end
        end

        return _blockprolong(field, block, basis, Val(O), 
            (cell..., cindex), 
            (stencils..., vertex_value_stencil(basis, Val(O + 1), Val(O + 1))), 
        rest...)
    end
end

@generated function _blockprolong(field::Field{N, T}, block::Block{N, T}, basis::AbstractBasis{T}, ::Val{O}, cell::NTuple{L, Int}, stencils::NTuple{L, AbstractStencil}, index::SubCellIndex, rest::Vararg{PointIndex, M}) where {N, T, O, L, M}
    side_expr = side -> quote
        ctotal = blockcells(block)[$L]
        cindex = vertex_to_cell(index)

        leftcells = min($O, cindex)
        rightcells = min($O, total - cindex)

        # Left side
        if leftcells < $O
            Base.@nexprs $O i -> begin
                if leftcells == i - 1
                    return _blockprolong(field, block, basis, Val(O), 
                        (cell..., cindex), 
                        (stencils..., subcell_value_stencil(basis, Val($side), Val(i - 1), Val(2* $O))), 
                    rest...)
                end
            end
        end

        # Right side
        if rightcells < $O
            Base.@nexprs $O i -> begin
                if rightcells == i - 1
                    return _blockprolong(field, block, basis, Val(O), 
                        (cell..., cindex), 
                        (stencils..., subcell_value_stencil(basis, Val($side), Val(2*$O), Val(i - 1))), 
                    rest...)
                end
            end
        end

        return _blockprolong(field, block, basis, Val(O), 
            (cell..., cindex), 
            (stencils..., subcell_value_stencil(basis, Val($side), Val(O), Val(O))), 
        rest...)
    end

    quote
        if subcell_side(index)
            $(side_expr(true))
        else
            $(side_expr(false))
        end
    end
end

#########################
## Domain ###############
#########################

export Domain, domainvalue, domainevaluate, domaingradient, domainhessian

"""
A block with fictious nodes filled.
"""
struct Domain{N, T, O}
    inner::Array{T, N}

    Domain{O}(inner::Array{T, N}) where {N, T, O} = new{N, T, O}(inner)
end

domaincells(domain::Domain{N, T, O}) where {N, T, O} = size(domain.inner) .- 2O
domainvalue(domain::Domain{N, T}, cell::CartesianIndex{N, T}) where {N, T} = domain.inner[CartesianIndex(cell.I .+ O)]
# setvalue!(domain::Domain{N, T}, value::T, cell::CartesianIndex{N, T}) where {N, T} = domain.inner[CartesianIndex(cell.I .+ O)] = value

"""
Apply the tensor product of a set of stencils at a point on a numerical domain
"""
function domain_stencil_product(domain::Domain{N, T}, cell::CartesianIndex{N}, stencils::NTuple{N, AbstractStencil{T}}) where {N, T}
    _domain_stencil_product(domain, cell, stencils...)
end

function _domain_stencil_product(domain::Domain{N, T}, cell::CartesianIndex{N}) where {N, T, L}
    domainvalue(domain, cell)
end

function _domain_stencil_product(domain::Domain{N, T}, cell::CartesianIndex{N}, stencil::CellStencil{T}, rest::Vararg{AbstractStencil{T}, L}) where {N, T, L}
    axis = N - L
    result = stencil.center * _domain_stencil_product(domain, cell, rest...)

    for (i, left) in enumerate(stencil.left)
        offcell = CartesianIndex(setindex(cell, cell[axis] - i, axis))
        result += left * _domain_stencil_product(domain, offcell, rest...)
    end

    for (i, right) in enumerate(stencil.right)
        offcell = CartesianIndex(setindex(cell, cell[axis] + i, axis))
        result += right * _domain_stencil_product(domain, offcell, rest...)
    end

    result
end

function _domain_stencil_product(domain::Domain{N, T}, cell::CartesianIndex{N}, stencil::VertexStencil{T}, rest::Vararg{AbstractStencil{T}, L}) where {N, T, L}
    axis = N - L
    result = zero(T)

    for (i, left) in enumerate(stencil.left)
        offcell = CartesianIndex(setindex(cell, cell[axis] + 1 - i, axis))
        result += left * _domain_stencil_product(domain, offcell, rest...)
    end

    for (i, right) in enumerate(stencil.right)
        offcell = CartesianIndex(setindex(cell, cell[axis] + i, axis))
        result += right * _domain_stencil_product(domain, offcell, rest...)
    end

    result
end

# Domain evaluation

"""
Evaluates the tensor product of a set of abstract operators at a point on a domain.
"""
function domainevaluate(domain::Domain{N, T, O}, cell::CartesianIndex{N}, basis::AbstractBasis{T}, opers::NTuple{N, AbstractOperator}) where {N, T, O}
    stencils = map(i -> operator_stencil(basis, Val(O), i), opers)
    domain_stencil_product(domain, cell, stencils)
end

# Domain Prolongation

"""
Performs prolongation for a full domain.
"""
function domainprolong(domain::Domain{N, T, O}, point::NTuple{N, PointIndex}, basis::AbstractBasis{T}) where {N, T, O}
    cell = CartesianIndex(map(point_to_cell, point))
    stencils = map(i -> _point_to_prolong_stencil(Val(O), i, basis), point) 
    domain_stencil_product(domain, cell, stencils)
end

function _point_to_prolong_stencil(::Val{O}, ::CellIndex, basis::AbstractBasis{T}) where {T, O}
    cell_value_stencil(basis, Val(O), Val(O))
end

function _point_to_prolong_stencil(::Val{O}, ::VertexIndex, basis::AbstractBasis{T}) where {T, O}
    vertex_value_stencil(basis, Val(O + 1), Val(O + 1))
end

function _point_to_prolong_stencil(::Val{O}, index::SubCellIndex, basis::AbstractBasis{T}) where {T, O}
    if subcell_side(index)
        return subcell_value_stencil(basis, Val(true), Val(O), Val(O))
    else
        return subcell_value_stencil(basis, Val(false), Val(O), Val(O))
    end
end

#########################
## Functional ###########
#########################

"""
Computes the gradient at a cell on a domain.
"""
function domaingradient(domain::Domain{N, T, O}, cell::CartesianIndex{N}, basis::AbstractBasis{T}) where {N, T, O}
    cells = size(domain.inner) .- 2O

    SVector(
        ntuple(Val(N)) do i
            opers = ntuple(dim -> ifelse(i == dim, ValueOperator{1}(), ValueOperator{0}()), Val(N))
            return domainevaluate(domain, cell, basis, opers) * cells[i]
        end
    ) 
end

"""
Computes the hessian at a cell on a domain.
"""
function domainhessian(domain::Domain{N, T, O}, cell::CartesianIndex{N}, basis::AbstractBasis{T}) where {N, T, O}
    cells = size(domain.inner) .- 2O

    hess = ntuple(Val(N * N)) do index
        i = (index - 1) ÷ N + 1
        j = (index - 1) % N + 1

        if i == j
            opers = ntuple(dim -> ifelse(i == dim, ValueOperator{2}(), ValueOperator{0}()), Val(N))
            return domainevaluate(domain, cell, basis, opers) * cells[i]^2
        else
            opers = ntuple(dim -> ifelse(i == dim || j == dim, ValueOperator{1}(), ValueOperator{0}()), Val(N))
            return domainevaluate(domain, cell, basis, opers) * cells[i] * cells[j]
        end
    end

    SMatrix{N, N, T}(hess)
end

#########################
## Evaluation ###########
#########################

export transfer_block_to_domain!, fill_interface!, interface_condition

"""
Computes the value the boundary should take at the cell and specific vertex
"""
interface_condition(field::Field{N, T}, block::Block{N, T}, cell::CartesianIndex{N}, basis::AbstractBasis{T}, ::Val{O}, vertex::NTuple{N, Int}) where {N, T, O} = error("Unimplemented")

"""
Transfers the data of a field on a block to a domain. This is essentally a preprocessing step which allows
all subsequent operations on the domain to be much cheaper.
"""
function transfer_block_to_domain!(domain::Domain{N, T, O}, field::Field{N, T}, block::Block{N, T}, basis::AbstractBasis{T}) where {N, T, O}
    # Meta data
    cells = blockcells(block)

    # Resize domain if necessary
    domaincells = cells .+ 2O
    if size(domain) != domaincells
        domain = Domain{N, T, O}(Array{T}(undef, domaincells...))
    end

    # Fill interior values 
    for cell in cellindices(block)
        v = value(field, block, cell)
        domain.inner[CartesianIndex(cell.I .+ O)] = v
    end

    # Fill exterior values
    _fill_interfaces!(domain, field, block, basis)
end

@generated function _fill_interfaces!(domain::Domain{N, T, O}, field::Field{N, T}, block::Block{N, T}, basis::AbstractBasis{T}) where {N, T, O}
    Base.@nexprs O i -> begin
        for subdomain in CartesianIndex(ntuple(_ -> 3, i))
            if sum(subdomain.I .== 1 .|| subdomain.I .== 3) == i
                fill_interface!(domain, field, block, basis, subdomain.I .- 2)
            end
        end
    end
end

function fill_interface!(domain::Domain{N, T, O}, field::Field{N, T}, block::Block{N, T}, basis::AbstractBasis{T}, subdomain::NTuple{N, Int}) where {N, T, O}
    cells = blockcells(block)

    facecells = ntuple(i -> ifelse(subdomain[i] == 0, 2:(cells[i] - 1), 1:1), Val(N))
    exteriorcells = ntuple(i -> ifelse(subdomain[i] ≠ 0, O, 1), Val(N))

    for facecell in CartesianIndices(facecells)
        cell = ntuple(Val(N)) do i
            if subdomain[i] == 1
                cells[i]
            elseif subdomain[i] == -1
                1
            else
                facecell[i]
            end
        end

        interface_value = interface_condition(field, block, CartesianIndex(cell), basis, Val(O), subdomain)
        
    end
end