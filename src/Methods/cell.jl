###############
## Exports ####
###############

export Face, cellface, boundaryface
export Cell, position, jacobian
export Field, value, gradient, hessian

###############
## Elements ###
###############

"""
A face between two `Element`s.
"""
const cellface::Int = 0
"""
A face between an `Element` and boundary.
"""
const boundaryface::Int = 1

"""
A face in a mesh.
"""
struct Face
    # Indicates what type of face this is.
    kind::Int
    # An index, either into the boundary array or the elements vector
    index::Int
end

"""
Represents a cell in a mesh. Eventually this will be extended to arbitrary 
hyper-quadralaterials, but for now it is simply a hyperrectangle with a certain
width and origin.
"""
struct Cell{N, T}
    # Bounds of the cell
    bounds::HyperBox{N, T}
    # Faces of this cell
    faces::NTuple{N, NTuple{2, Face}}
    # Number of dofs along each dimension
    dofs::SVector{N, Int}
    # Refinement
    refinement::Int
    # Transform
    local_to_global::ComposedTransform{N, T, Translate{N, T}, ScaleTransform{N, T}}

    function Cell(bounds::HyperBox{N, T}, faces::NTuple{N, NTuple{2, Face}}, dofs::SVector{N, Int}) where {N, T}
        scales::SVector{N, T} = bounds.widths ./ dofs
        local_to_global = Translate(bounds.origin) ∘ ScaleTransform(scales)

        new{N, T}(bounds, faces, dofs, 0, local_to_global)
    end
end

Base.length(cell::Cell) = prod(cell.dofs)
Base.eachindex(cell::Cell) = CartesianIndices(Tuple(cell.dofs))

position(cell::Cell{N, T}, point::CartesianIndex{N}) where {N, T} = cell.local_to_global(SVector{N, T}(point.I))
jacobian(cell::Cell{N, T}, point::CartesianIndex{N}) where {N, T} = jacobian(cell.local_to_global, SVector{N, T}(point.I))

"""
The values of a field on an cell.
"""
struct Field{N, T} 
    values::Array{T, N}
end

value(::Cell{N, T}, point::CartesianIndex{N}, field::Field{N, T}) where {N, T} = field.values[point]

# Generic Functions

function stencil_with_offset(field::Field{N, T}, point::CartesianIndex{N}, di::Int, D::SBPDerivative{T, O}) where {N, T, O}
    dimoffset = size(field)[di] + 1

    if point[di] ≤ length(D.left_block)
        coefficientrow = point[di]
        stenciloff = 0
        stencil = D.left_block[coefficientrow]
    elseif point[di] ≥ dimoffset - length(D.right_block)
        coefficientrow = point[di] - (dimoffset - length(D.right_block)) + 1
        stenciloff = dimoffset - length(D.right_block)
        stencil = D.right_block[coefficientrow]
    else
        stenciloff = point[di] - O - 1
        stencil = D.central_coefs
    end

    return stencil, stenciloff
end

function derivative(field::Field{N, T}, point::CartesianIndex{N}, di::Int, D::SBPDerivative{T, O}) where {N, T, O}
    stencil, stenciloff = stencil_with_offset(field, point, di, D)

    result = zero(T)

    for i in eachindex(stencil)
        off = ntuple(Val(N)) do p
            ifelse(p == di, stenciloff + i, point[p])
        end
        result += stencil[i] * field.values[off...]
    end

    result
end

function derivative_mixed(field::Field{N, T}, point::CartesianIndex{N}, di::Int, dj::Int, D::SBPDerivative{T, O}) where {N, T, O}
    stencili, stenciloffi = stencil_with_offset(field, point, di, D)
    stencilj, stenciloffj = stencil_with_offset(field, point, dj, D)

    result = zero(T)

    for i in eachindex(stencili)
        offi = ntuple(Val(N)) do p
            ifelse(p == di, stenciloffi + i, point[p])
        end

        for j in eachindex(stencilj)
            off = ntuple(Val(N)) do p
                ifelse(p == dj, stenciloffj + j, offi[p])
            end

            result += stencili[i] * stencilj[j] * field.values[off...]
        end
    end

    result
end

# Gradient
function gradient(cell::Cell{N, T}, point::CartesianIndex{N}, field::Field{N, T}, derivatives::SBPDerivatives{T, O}) where {N, T, O}
    grad = ntuple(Val(N)) do dim
        derivative(field, point, dim, derivatives[1])
    end

    jacobian(cell, point) * SVector(grad)
end

# Second derivatives
function hessian_component(field::Field{N, T}, point::CartesianIndex{N}, di::Int, dj::Int, derivatives::SBPDerivatives{T}) where {N, T}
    if di == dj
        return derivative(field, point, di, derivatives[2])
    else
        return derivative_mixed(field, point, di, dj, derivatives[1])
    end
end

function hessian(cell::Cell{N, T}, point::CartesianIndex{N}, field::Field{N, T}, derivatives::SBPDerivatives{T}) where {N, T}
    hess = StaticArrays.sacollect(SMatrix{N, N, T, N^2}, hessian_component(field, point, i, j, derivatives) for i in 1:N, j in 1:N)
    j = jacobian(cell, point)
    j' * hess * j
end