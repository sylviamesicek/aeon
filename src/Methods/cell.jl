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
Produces a view into a field, reinterpreted as a multidimensional array.
"""
function cellfield(cell::Cell{N, T}, field::AbstractVector{T}) where {N, T}
    total = length(cell)

    subfield = @view field[1:total .+ cell.dofoffset]

    reshape(subfield, cell.dofs)
end

"""
Returns a tuple of the faces of the cell.
"""
function cellfaces(cell::Cell{N, T}) where {N, T}
    ntuple(Val(2*N)) do i
        if i > N
            return Face(i - N, true, cell.faces[i - N][2])
        else
            return Face(i, false, cell.faces[i][1])
        end
    end
end

# """
# Converts a face side into its index.
# """
# face_side_to_index(cell::Cell{N, T}, face::Face) where {N, T} =  face.side ? cell.dofs[face.axis] : 1

# face_opposite_side_to_index(cell::Cell{N, T}, face::Face) where {N< T} = face.side ? 1 : cell.dofs[face.axis]

cell_dofs_off_axis(cell::Cell{N, T}, ::Val{A}) where {N, T, A} = tuple(cell.dofs[begin:(A-1)]..., cell.dofs[(A+1):end])

project_on_axis(index::CartesianIndex, value, ::Val{A}) where {A} = CartesianIndex(tuple(index.I[begin:(A-1)]..., value, index.I[(A+1):end]))

function face_indices(cell::Cell{N, T}, ::Val{A}) where {N, T, A}
    facedofs = cell_dofs_off_axis(cell, Val(A))

    CartesianIndices(facedofs)
end

"""
Returns the meta data associated with a face of a cell. This may be further interpreted by `isboundary` and `isinterface`.
"""
function face_meta(cell::Cell, face::Face)
    abs(cell[face.axis][face.side])
end

"""
Returns true if the given face lies between the cell and a boundary of the domain, false otherwise.
"""
function face_isboundary(cell::Cell, face::Face)
    cell[face.axis][face.side] ≤ 0
end

"""
Returns true if the given face lies between two cells, false otherwise.
"""
function face_isinterface(cell::Cell, face::Face)
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


##########################
## Interfaces ############
##########################

function smooth_interface_left_to_right!(result::AbstractVector{T}, prolong::ProlongationOperator{T}, restrict::RestrictionOperator{T}, x::AbstractVector{T}, 
    left::Cell{N, T}, ::Val{A}, right::Cell{N, T}, strength::T) where {N, T, A}

    # Compute views into functions
    leftx = cellfield(left, x)
    rightx = cellfield(right, x)
    leftresult = cellfield(left, result)

    if left.refinement == right.refinement
        for index in face_indices(left, Val(A))
            leftindex = project_on_axis(index, left.dofs[A], A)
            rightindex = project_on_axis(index, 1, A)

            leftresult[leftindex] += strength * (leftx[leftindex] - rightx[rightindex])
        end
    elseif left.refinement > right.refinement
        diritchlet_prolong = ntuple(dim -> dim == A ? prolong : IdentityOperator(), Val(N))

        for index in face_indices(left, Val(A))
            leftindex = project_on_axis(index, left.dofs[A], A)
            rightindex = project_on_axis(index, 1, A)

            value = product(rightindex, diritchlet_prolong, rightx)

            leftresult[leftindex] += strength * (leftx[leftindex] - value)
        end 
    else
        diritchlet_restrict = ntuple(dim -> dim == A ? restrict : IdentityOperator(), Val(N))

        for index in face_indices(left, Val(A))
            leftindex = project_on_axis(index, left.dofs[A], A)
            rightindex = project_on_axis(index, 1, A)

            value = product(rightindex, diritchlet_restrict, rightx)

            leftresult[leftindex] += strength * (leftx[leftindex] - value)
        end
    end
end

function smooth_interface_right_to_left!(result::AbstractVector{T}, prolong::ProlongationOperator{T}, restrict::RestrictionOperator{T}, x::AbstractVector{T}, 
    left::Cell{N, T}, ::Val{A}, right::Cell{N, T}, strength::T) where {N, T, A}

    # Compute views into functions
    leftx = cellfield(left, x)
    rightx = cellfield(right, x)
    rightresult = cellfield(left, result)

    if left.refinement == right.refinement
        for index in face_indices(right, Val(A))
            leftindex = project_on_axis(index, left.dofs[A], A)
            rightindex = project_on_axis(index, 1, A)

            rightresult[rightindex] += strength * (rightx[rightindex] - leftx[leftindex])
        end
    elseif left.refinement < right.refinement
        diritchlet_prolong = ntuple(dim -> dim == A ? prolong : IdentityOperator(), Val(N))

        for index in face_indices(right, Val(A))
            leftindex = project_on_axis(index, left.dofs[A], A)
            rightindex = project_on_axis(index, 1, A)

            value = product(leftindex, diritchlet_prolong, leftx)

            rightresult[rightindex] += strength * (rightx[rightindex] - value)
        end 
    else
        diritchlet_restrict = ntuple(dim -> dim == A ? restrict : IdentityOperator(), Val(N))

        for index in face_indices(right, Val(A))
            leftindex = project_on_axis(index, left.dofs[A], A)
            rightindex = project_on_axis(index, 1, A)
            
            value = product(leftindex, diritchlet_restrict, leftx)

            rightresult[rightindex] += strength * (rightx[rightindex] - value)
        end
    end
end