###############
## Exports ####
###############

# export Face, cellface, boundaryface
export Cell, position, jacobian
export value, gradient, hessian, laplacian

###############
## Elements ###
###############

# """
# A face between two `Element`s.
# """
# const cellface::Int = 0
# """
# A face between an `Element` and boundary.
# """
# const boundaryface::Int = 1

# """
# A face in a mesh.
# """
# struct Face
#     # Indicates what type of face this is.
#     kind::Int
#     # An index, either into the boundary array or the elements vector
#     index::Int
# end

"""
Represents a face of a cell. Contains an `axis` (the axis running perpendicular to the face), a side boolean which
indicates if the face is on the positive or negative side of the cell.
"""
struct Face
    axis::Int
    side::Bool
end

Base.:(-)(face::Face) = Face(face.axis, !face.side)

"""
Represents a cell in a mesh. Eventually this will be extended to arbitrary 
hyper-quadralaterials, but for now it is simply a hyperrectangle with a certain
width and origin.
"""
struct Cell{N, T}
    # Bounds of the cell
    bounds::HyperBox{N, T}
    # Faces of this cell
    faces::NTuple{N, NTuple{2, Int}}
    # Number of dofs along each dimension
    dofs::SVector{N, Int}
    # Refinement
    refinement::Int
    # Transform
    local_to_global::ComposedTransform{N, T, Translate{N, T}, ScaleTransform{N, T}}
    # Offset into dof vector.
    dofoffset::Int

    function Cell(bounds::HyperBox{N, T}, faces::NTuple{N, NTuple{2, Int}}, basedofs::SVector{N, Int}, refinement::Int, dofoffset::Int) where {N, T}
        dofs = 2 .^ (basedofs .+ refinement) .+ 1

        scales::SVector{N, T} = bounds.widths ./ (dofs .- 1)
        local_to_global = Translate(bounds.origin) ∘ ScaleTransform(scales)

        new{N, T}(bounds, faces, dofs, refinement, local_to_global, dofoffset)
    end
end

Base.length(cell::Cell) = prod(cell.dofs)
Base.eachindex(cell::Cell) = CartesianIndices(Tuple(cell.dofs))

"""
Returns a tuple of the faces of the cell.
"""
function faces(cell::Cell{N, T}) where {N, T}
    ntuple(Val(2*N)) do i
        if i > N
            return Face(i - N, true, cell.faces[i - N][2])
        else
            return Face(i, false, cell.faces[i][1])
        end
    end
end

axis_face_index(cell::Cell{N, T}, face::Face) where {N, T} =  face.side ? cell.dofs[face.axis] : 1

"""
Iterates the points that lie on a given face of the cell.
"""
function eachindexface(cell::Cell{N, T}, face::Face) where {N, T}
    facedofs = ntuple(Val(N - 1)) do dim
        cell.dofs[dim + (dim ≥ face.axis)]
    end

    faceindices = CartesianIndices(facedofs)

    Iterators.map(faceindices) do index
        fullindex = ntuple(Val(N)) do dim
            dim == axis ? axis_face_index(cell, face) : index[dim - (dim > axis)]
        end

        CartesianIndex(fullindex)
    end
end

"""
Returns the index associated with a face of a cell. This may be further interpreted by `isboundary` and `isinterface`.
"""
function faceid(cell::Cell{N, T}, face::Face)
    abs(cell[face.axis][face.side])
end

"""
Returns true if the given face lies between the cell and a boundary of the domain, false otherwise.
"""
function isboundary(cell::Cell{N, T}, face::Face)
    cell[face.axis][face.side] ≤ 0
end

"""
Returns true if the given face lies between two cells, false otherwise.
"""
function isinterface(cell::Cell{N, T}, face::Face)
    cell[face.axis][face.side] > 0
end

"""
Computes the global position of a point in a cell.
"""
position(cell::Cell{N, T}, point::CartesianIndex{N}) where {N, T} = cell.local_to_global(SVector{N, T}(Tuple(point) .- 1))

"""
Computes the jacobian at the point in a cell.
"""
jacobian(cell::Cell{N, T}, point::CartesianIndex{N}) where {N, T} = jacobian(cell.local_to_global, SVector{N, T}(Tuple(point) .- 1))

"""
Computes the global dof index of a point.
"""
function local_to_global(cell::Cell{N, T}, point::CartesianIndex{N}) where {N, T}
    linear = LinearIndices(Tuple(cell.dofs))
    cell.dofoffset + linear[point]
end

# Helper function
function local_to_global(cell::Cell{N, T}, point::NTuple{N, Int}) where {N, T}
    linear = LinearIndices(Tuple(cell.dofs))
    cell.dofoffset + linear[point...]
end

########################
## Operators ###########
########################

"""
Computes the value of a field at a point in a cell.
"""
value(cell::Cell{N, T}, point::CartesianIndex{N}, field::AbstractVector{T}) where {N, T} = field[local_to_global(cell, point)]

function stencil_with_offset(cell::Cell{N, T}, point::CartesianIndex{N},  di::Int, operator::SBPOperator{T}, rank::Int) where {N, T}
    dimoffset = cell.dofs[di] + 1

    D = derivative(operator, rank)

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

function approx_derivative(cell::Cell{N, T}, point::CartesianIndex{N}, field::AbstractVector{T}, di::Int, operator::SBPOperator{T}, rank::Int) where {N, T}
    stencil, stenciloff = stencil_with_offset(cell, point, di, operator, rank)

    result = zero(T)

    for i in eachindex(stencil)
        ptrlocal = ntuple(Val(N)) do p
            ifelse(p == di, stenciloff + i, point[p])
        end

        ptrglobal = local_to_global(cell, ptrlocal)

        result += stencil[i] * field[ptrglobal]
    end

    result
end

function approx_derivative_mixed(cell::Cell{N, T}, point::CartesianIndex{N}, field::AbstractVector{T}, di::Int, dj::Int, operator::SBPOperator{T}, rank::Int) where {N, T}
    stencili, stenciloffi = stencil_with_offset(cell, point, di, operator, rank)
    stencilj, stenciloffj = stencil_with_offset(cell, point, dj, operator, rank)

    result = zero(T)

    for i in eachindex(stencili)
        ptrlocali = ntuple(Val(N)) do p
            ifelse(p == di, stenciloffi + i, point[p])
        end

        for j in eachindex(stencilj)
            ptrlocal = ntuple(Val(N)) do p
                ifelse(p == dj, stenciloffj + j, ptrlocali[p])
            end

            ptrglobal = local_to_global(cell, ptrlocal)

            result += stencili[i] * stencilj[j] * field[ptrglobal]
        end
    end

    result
end

# Gradient
"""
Computes the gradient of a field at a point in a cell, using the given operator.
"""
function gradient(cell::Cell{N, T}, point::CartesianIndex{N}, field::AbstractVector{T}, operator::SBPOperator{T}) where {N, T}
    grad = ntuple(Val(N)) do dim
        approx_derivative(cell, point, field, dim, operator, 1)
    end

    jacobian(cell, point) * SVector(grad)
end

# Second derivatives
function hessian_component(cell::Cell{N, T}, point::CartesianIndex{N}, field::AbstractVector{T}, di::Int, dj::Int, operator::SBPOperator{T}) where {N, T}
    if di == dj
        return approx_derivative(cell, point, field, di, operator, 2)
    else
        return approx_derivative_mixed(cell, point, field, di, dj, operator, 1)
    end
end

"""
Computes the hessian of a field at a point in a cell using the given operator.
"""
function hessian(cell::Cell{N, T}, point::CartesianIndex{N}, field::AbstractVector{T}, operator::SBPOperator{T}) where {N, T}
    hess = StaticArrays.sacollect(SMatrix{N, N, T, N^2}, hessian_component(cell, point, field, i, j, operator) for i in 1:N, j in 1:N)
    j = jacobian(cell, point)
    j' * hess * j
end

@generated function laplacian(cell::Cell{N, T}, point::CartesianIndices{N}, field::AbstractVector{T}, operator::SBPOperator{T}) where {N, T}
    components = [:(hess[$i, $i]) for i in 1:N]
    result = Expr(:call, :+, components...)

    quote
        hess = hessian(cell, point, field, operator)
        $(result)
    end
end


##########################
## Interfaces ############
##########################

function smooth_interfaces!(result::AbstractVector{T}, x::AbstractVector{T}, mesh::Mesh{N, T}, cell::Cell{N, T}, strength::T) where {N, T}
    for face in faces(cell)
        if !isinterface(cell, face)
            continue
        end

        other = mesh[faceid(cell, face)]

        if cell.refinement == other.refinement
            for faceindex in eachindexface(cell, face)
                # Other index is the same with a reflected face.
                otherindex = CartesianIndex(ntuple(Val(N)) do dim
                    dim == face.axis ? axis_face_index(other, -face) : faceindex[dim]
                end)

                cellptr = local_to_global(cell, faceindex)
                otherptr = local_to_global(other, otherindex)

                # Diritchlet interface conditions
                result[cellptr] += strength * (x[cellptr] - x[otherptr])
            end
        elseif cell.refinement < other.refinement
            for faceindex in eachindexface(cell, face)
                # Other index is scaled
                otherindex = CartesianIndex(ntuple(Val(N)) do dim
                    dim == face.axis ? axis_face_index(other, -face) : 2faceindex[dim] - 1
                end)

                cellptr = local_to_global(cell, faceindex)
                otherptr = local_to_global(other, otherindex)

                # Diritchlet interface conditions
                result[cellptr] += strength * (x[cellptr] - x[otherptr])
            end
        else
            # for faceindex in eachindexface(cell, face)
            #     # Other index is scaled
            #     otherindex = CartesianIndex(ntuple(Val(N)) do dim
            #         dim == face.axis ? axis_face_index(other, -face) : 2faceindex[dim] - 1
            #     end)

            #     cellptr = local_to_global(cell, faceindex)
            #     otherptr = local_to_global(other, otherindex)

            #     # Diritchlet interface conditions
            #     result[cellptr] += strength * (x[cellptr] - x[otherptr])
            # end
        end
    end
end