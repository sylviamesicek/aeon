export PointIndex, CellIndex, SubCellIndex, VertexIndex
export Block, blockcells, blockbounds, blocktransform, blockprolong
export cellindices, cellwidths, cellcenter
export Field, value, setvalue!
export Domain, domainvalue, domaingradient, domainhessian
export domainevaluate, domainprolong
export fill_interface!, transfer_block_to_domain!

#####################
## Points ###########
#####################

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

"""
Computes the cell of which this subcell is a subdivision.
"""
subcell_to_cell(v::SubCellIndex) = CellIndex((v.inner + 1) ÷ 2)

"""
A vertex index.
"""
struct VertexIndex <: PointIndex
    inner::Int
end

# Gets the cell to the left of this vertex
vertex_to_cell(v::VertexIndex) = CellIndex(v.inner - 1)

########################
## Block ###############
########################

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
value(field::Field{N, T}, block::Block{N, T}, cell::CartesianIndex{N, T}) where {N, T} = error("Unimplemented")

"""
Sets the value of a field at a cell in a block (must be implemented for each field type).
"""
setvalue!(field::Field{N, T}, value::T, block::Block{N, T}, cell::CartesianIndex{N, T}) where {N, T} = error("Unimplemented")

#############################
## Prolongation (Block) #####
#############################

"""
Performs prolongation within a block, to the given order.
"""
function blockprolong(field::Field{N, T}, block::Block{N, T}, point::NTuple{N, PointIndex}, basis::AbstractBasis{T}, ::Val{O}) where {N, T, O}
    _prolong_product(field, block, basis, Val(O), (), point...)
end

function blockprolong(field::Field{N, T}, block::Block{N, T}, cell::NTuple{N, CellIndex}, ::AbstractBasis{T}, ::Val{O}) where {N, T, O}
    value(field, block, CartesianIndex(map(i -> i.inner, cell)))
end

function _prolong_product(field::Field{N, T}, block::Block{N, T}, basis::AbstractBasis{T}, ::Val{O}, cell::NTuple{N, CellIndex}) where {N, T, O}
    blockprolong(field, block, cell, basis, Val(O))
end

function _prolong_product(field::Field{N, T}, block::Block{N, T}, basis::AbstractBasis{T}, ::Val{O}, cell::NTuple{M, CellIndex}, index::CellIndex, rest::Vararg{PointIndex, L}) where {N, T, L, M, O}
    _prolong_product(field, block, basis, Val(O), (cell..., index), rest...)
end

@generated function _prolong_product(field::Field{N, T}, block::Block{N, T}, basis::AbstractBasis{T}, ::Val{O}, cell::NTuple{M, CellIndex}, index::VertexIndex, rest::Vararg{PointIndex, L}) where {N, T, L, M, O}
    quote
        ctotal = blockcells(block)[$L]
        cindex = vertex_to_cell(index)

        leftcells = min($O, cindex)

        # Left side
        if leftcells < $O
            Base.@nexprs $O i -> begin
                if leftcells == i - 1
                    return _prolong_vertex(field, block, vertex_value_stencil(basis, Val(i - 1), Val(2 * $O + 1)), basis, Val($O), cell, index, rest...)
                end
            end
        end

        rightcells = min($O, total - cindex)

        # Right side
        if rightcells < $O
            Base.@nexprs $O i -> begin
                if rightcells == i - 1
                    return _prolong_vertex(field, block, vertex_value_stencil(basis, Val(2 * $O + 1), Val(i - 1)), basis, Val($O), cell, index, rest...)
                end
            end
        end

        return _prolong_vertex(field, block, vertex_value_stencil(basis, Val($O + 1), Val($O + 1)), basis, Val($O), cell, index, rest...)
    end
end

@generated function _prolong_product(field::Field{N, T}, block::Block{N, T}, basis::AbstractBasis{T}, ::Val{O}, cell::NTuple{M, CellIndex}, index::SubCellIndex, rest::Vararg{PointIndex, L}) where {N, T, L, M, O}
    side_expr = side -> quote
        # Left side
        if leftcells < $O
            Base.@nexprs $O i -> begin
                if leftcells == i - 1
                    return _prolong_subcell(field, block, subcell_value_stencil(basis, Val($side), Val(i - 1), Val(2O)), basis, Val($O), cell, index, rest...)
                end
            end
        end
        # Right side
        if rightcells < $O
            Base.@nexprs $O i -> begin
                if rightcells == i - 1
                    return _prolong_subcell(field, block, subcell_value_stencil(basis, Val($side), Val(2O), Val(i - 1)), basis, Val($O), cell, index, rest...)
                end
            end
        end

        return _prolong_subcell(field, block, subcell_value_stencil(basis, Val($side), Val(O), Val(O)), basis, Val($O), cell, index, rest...)
    end
    
    quote
        ctotal = blockcells(block)[$L]
        cindex = subcell_to_cell(index)

        leftcells = min($O, cindex - 1)
        rightcells = min($O, total + 1 - cindex)

        if subcell_side(index)
            $(side_expr(true))
        else
            $(side_expr(false))
        end
    end
end

"""
Applies a vertex stencil to a field in a block such that `_prolong_product` is recursively called for the remaining indices.
"""
function _prolong_vertex(field::Field{N, T}, block::Block{N, T}, stencil::VertexStencil{T}, basis::AbstractBasis{T}, ::Val{O}, cell::NTuple{M, CellIndex}, index::VertexIndex, rest::Vararg{PointIndex, L}) where {N, T, L, M, O}
    result = zero(T)

    cindex = vertex_to_cell(index)

    for (i, left) in enumerate(stencil.left)
        offcell = cindex + 1 - i
        result += left * _prolong_product(field, block, basis, Val(O), (cell..., offcell), rest...)
    end

    for (i, right) in enumerate(stencil.right)
        offcell = cindex + i
        result += right * _prolong_product(field, block, basis, Val(O), (cell..., offcell), rest...)
    end
    
    return result
end

"""
Applies a subcell stencil to a field in a block such that `_prolong_product` is recursively called for the remaining indices.
"""
function _prolong_subcell(field::Field{N, T}, block::Block{N, T}, stencil::SubCellStencil{T}, basis::AbstractBasis{T}, ::Val{O}, cell::NTuple{M, CellIndex}, index::SubCellIndex, rest::Vararg{PointIndex, L}) where {N, T, L, M, O}
    cindex = subcell_to_cell(index)
    result = stencil.center * _prolong_product(field, block, basis, Val(O), (cell..., cindex), rest...)

    for (i, left) in enumerate(stencil.left)
        offcell = cindex - i
        result += left * _prolong_product(field, block, basis, Val(O), (cell..., offcell), rest...)
    end

    for (i, right) in enumerate(stencil.right)
        offcell = cindex + i
        result += right * _prolong_product(field, block, basis, Val(O), (cell..., offcell), rest...)
    end
    
    return result
end

#########################
## Domain ###############
#########################

"""
A block with fictious nodes filled.
"""
struct Domain{N, T, O}
    inner::Array{T, N}

    Domain{O}(inner::Array{T, N}) where {N, T, O} = new{N, T, O}(inner)
end

domainvalue(domain::Domain{N, T}, cell::CartesianIndex{N, T}) where {N, T} = domain.inner[CartesianIndex(cell.I .+ O)]
# setvalue!(domain::Domain{N, T}, value::T, cell::CartesianIndex{N, T}) where {N, T} = domain.inner[CartesianIndex(cell.I .+ O)] = value

# Domain Prolongation

"""
Performs prolongation for a full domain.
"""
function domainprolong(domain::Domain{N, T, O}, point::NTuple{N, PointIndex}, basis::AbstractBasis{T}) where {N, T, O}
    _prolong_product(domain, basis, (), point...)
end

function domainprolong(domain::Domain{N, T, O}, cell::NTuple{N, CellIndex}, ::AbstractBasis{T}) where {N, T, O}
    domainvalue(domain, CartesianIndex(map(i -> i.inner, cell .+ O)))
end

function domainprolong(domain::Domain{N, T, O}, cell::CartesianIndex{N}, ::AbstractBasis{T}) where {N, T, O}
    domainvalue(domain, map(CellIndex, cell.I))
end

# Product

function _prolong_product(domain::Domain{N, T, O}, basis::AbstractBasis{T}, cell::NTuple{N, CellIndex}, index::CellIndex, rest::Vararg{PointIndex, L}) where {N, T, L}
    _prolong_product(domain, basis, (cell..., index), rest)
end

function _prolong_product(domain::Domain{N, T, O}, basis::AbstractBasis{T}, cell::NTuple{N, CellIndex}, index::VertexIndex, rest::Vararg{PointIndex, L}) where {N, T, L}
    cindex = vertex_to_cell(index)
    stencil = vertex_value_stencil(basis, Val(O + 1), Val(O + 1))
    result = zero(T)

    for (i, left) in enumerate(stencil.left)
        offcell = cindex + 1 - i
        result += left * _prolong_product(domain, basis, (cell..., offcell), rest...)
    end

    for (i, right) in enumerate(stencil.right)
        offcell = cindex + i
        result += right * _prolong_product(domain, basis, (cell..., offcell), rest...)
    end
    
    return result
end

function _prolong_product(domain::Domain{N, T, O}, basis::AbstractBasis{T}, cell::NTuple{N, CellIndex}, index::SubCellIndex, rest::Vararg{PointIndex, L}) where {N, T, L}
    cindex = subcell_to_cell(index)

    if subcell_side(index)
        stencil = subcell_value_stencil(basis, Val(true), Val(O), Val(O))
    else
        stencil = subcell_value_stencil(basis, Val(false), Val(O), Val(O))
    end

    result = stencil.center * _prolong_product(domain, basis, (cell..., cindex), rest...)

    for (i, left) in enumerate(stencil.left)
        offcell = cindex - i
        result += left * _prolong_product(domain, basis, (cell..., offcell), rest...)
    end

    for (i, right) in enumerate(stencil.right)
        offcell = cindex + i
        result += right * _prolong_product(domain, basis, (cell..., offcell), rest...)
    end
    
    return result
end

# Domain evaluation

"""
Evaluates the tensor product of the given operators on a domain
"""
function domainevaluate(domain::Domain{N, T, O}, cell::NTuple{N, CellIndex}, basis::AbstractBasis{T}, opers::NTuple{N, AbstractOperator}) where {N, T, O}
    _evaluate_product(domain, cell, basis, opers...)
end

function domainevaluate(domain::Domain{N, T, O}, cell::CartesianIndex{N}, basis::AbstractBasis{T}, opers::NTuple{N, AbstractOperator}) where {N, T, O}
    domainevaluate(domain, map(CellIndex, cell.I), basis, opers)
end

# Product

function _evaluate_product(domain::Domain{N, T, O}, cell::NTuple{N, CellIndex}, basis::AbstractBasis{T}, oper::AbstractOperator, rest::Vararg{AbstractOperator, L}) where {N, T, O, L}
    stencil = operator_stencil(basis, Val(O), oper)
    
    result = stencil.center * _evaluate_product(domain, cell, basis, rest...)

    for (i, left) in enumerate(stencil.left)
        offcell = setindex(cell, cell[L] - i, L)
        result += left * _evaluate_product(domain, offcell, basis, rest...)
    end

    for (i, right) in enumerate(stencil.right)
        offcell = setindex(cell, cell[L] + i, L)
        result += right * _evaluate_product(domain, offcell, basis, rest...)
    end
    
    return result
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

"""
Fills edges of a domain subjection to various boundary conditions. This must be implemented for each field and block type.
"""
fill_interface!(domain::Domain{N, T}, field::Field{N, T}, block::Block{N, T}, subdomain::NTuple{N, Int}) where {N, T} = error("Unimplemented")

"""
Transfers the data of a field on a block to a domain. This is essentally a preprocessing step which allows
all subsequent operations on the domain to be much cheaper.
"""
function transfer_block_to_domain!(domain::Domain{N, T, O}, field::Field{N, T}, block::Block{N, T}) where {N, T, O}
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
    _fill_interfaces!(domain, field, block)
end

@generated function _fill_interfaces!(domain::Domain{N, T, O}, field::Field{N, T}, block::Block{N, T}) where {N, T, O}
    Base.@nexprs O i -> begin
        for subdomain in CartesianIndex(ntuple(_ -> 3, i))
            if sum(subdomain.I .== 1 .|| subdomain.I .== 3) == i
                fill_interface!(domain, field, block, subdomain.I .- 2)
            end
        end
    end
end