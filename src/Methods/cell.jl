###############
## Exports ####
###############

# export Face, cellface, boundaryface
export Cell, position, jacobian

###############
## Cells ######
###############

"""
Represents a cell in a mesh. Eventually this will be extended to arbitrary 
hyper-quadralaterials, but for now it is simply a hyperrectangle with a certain
width and origin. Connectivity (ie, which faces connect which cells) is stored within
the mesh object.
"""
struct Cell{N, T}
    # Bounds of the cell
    bounds::HyperBox{N, T}
    # Transform
    local_to_global::ComposedTransform{N, T, Translate{N, T}, ScaleTransform{N, T}}
    # Refinement
    refinement::Int
    # Offset into dof vector.
    dofoffset::Int

    function Cell(bounds::HyperBox{N, T}, refinement::Int, dofoffset::Int) where {N, T}
        scales::SVector{N, T} = bounds.widths ./ (dofs .- 1)
        local_to_global = Translate(bounds.origin) ∘ ScaleTransform(scales)

        new{N, T}(bounds, local_to_global, refinement, dofoffset)
    end
end

"""
Returns a tuple containing the number of dofs along each dimension. This is simply a function of the refinement level of the cell.
"""
celldofs(cell::Cell{N, T}) where {N, T} = ntuple(i -> 2^cell.refinement + 1, Val(N))

Base.length(cell::Cell{N}) where N = (2^cell.refinement + 1)^N
Base.eachindex(cell::Cell) = CartesianIndices(celldofs(cell))

"""
Produces a view into a field, reinterpreted as a multidimensional array.
"""
function cellfield(cell::Cell{N, T}, field::AbstractVector{T}) where {N, T}
    total = length(cell)

    subfield = @view field[1:total .+ cell.dofoffset]

    reshape(subfield, celldofs(cell))
end

function cell_indices_on_axis(cell::Cell{N, T}, ::Val{A}) where {N, T, A}
    CartesianIndices(tuple_project_on_axis(celldofs(cell), Val(A)))
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

function smooth_interface!(result::AbstractVector{T}, prolong::ProlongationOperator{T}, restrict::RestrictionOperator{T}, x::AbstractVector{T}, 
    left::Cell{N, T}, ::Val{A}, right::Cell{N, T}, strength::T) where {N, T, A}

    # Compute views into functions
    leftx = cellfield(left, x)
    rightx = cellfield(right, x)
    leftresult = cellfield(left, result)
    rightresult = cellfield(right, result)

    if left.refinement == right.refinement
        for index in cell_indices_on_axis(left, Val(A))
            leftindex = index_splice_on_axis(index, Val(A), left.dofs[A])
            rightindex = index_splice_on_axis(index, Val(A), 1)

            leftresult[leftindex] += strength * (leftx[leftindex] - rightx[rightindex])
            rightresult[rightindex] += strength * (rightx[rightindex] - leftx[leftindex])
        end
    elseif left.refinement > right.refinement
        diritchlet_prolong = ntuple(dim -> dim == A ? prolong : IdentityOperator(), Val(N))
        diritchlet_restrict = ntuple(dim -> dim == A ? restrict : IdentityOperator(), Val(N))

        for index in face_indices(left, Val(A))
            leftindex = index_splice_on_axis(index, Val(A), left.dofs[A])
            rightindex = index_splice_on_axis(index, Val(A), 1)

            value = product(rightindex, diritchlet_prolong, rightx)

            leftresult[leftindex] += strength * (leftx[leftindex] - value)
        end 

        for index in face_indices(right, Val(A))
            leftindex = index_splice_on_axis(index, Val(A), left.dofs[A])
            rightindex = index_splice_on_axis(index, Val(A), 1)
            
            value = product(leftindex, diritchlet_restrict, leftx)

            rightresult[rightindex] += strength * (rightx[rightindex] - value)
        end
    else
        diritchlet_prolong = ntuple(dim -> dim == A ? prolong : IdentityOperator(), Val(N))
        diritchlet_restrict = ntuple(dim -> dim == A ? restrict : IdentityOperator(), Val(N))

        for index in face_indices(right, Val(A))
            leftindex = index_splice_on_axis(index, Val(A), left.dofs[A])
            rightindex = index_splice_on_axis(index, Val(A), 1)

            value = product(leftindex, diritchlet_prolong, leftx)

            rightresult[rightindex] += strength * (rightx[rightindex] - value)
        end 

        for index in face_indices(left, Val(A))
            leftindex = index_splice_on_axis(index, Val(A), left.dofs[A])
            rightindex = index_splice_on_axis(index, Val(A), 1)

            value = product(rightindex, diritchlet_restrict, rightx)

            leftresult[leftindex] += strength * (leftx[leftindex] - value)
        end
    end
end