###############
## Exports ####
###############

# export Face, cellface, boundaryface
export Cell, position, jacobian, cellfield, local_to_global, celldofs

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
        scales = SVector(ntuple(i -> bounds.widths[i] / 2^refinement, Val(N)))
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

    subfield = @view field[(1:total) .+ cell.dofoffset]

    reshape(subfield, celldofs(cell))
end

"""
Computes the global dof index of a point.
"""
function local_to_global(cell::Cell{N, T}, point::CartesianIndex{N}) where {N, T}
    linear = LinearIndices(celldofs(cell))
    cell.dofoffset + linear[point]
end

# Helper function
function local_to_global(cell::Cell{N, T}, point::NTuple{N, Int}) where {N, T}
    linear = LinearIndices(celldofs(cell))
    cell.dofoffset + linear[point...]
end

# function cell_indices_on_axis(cell::Cell{N, T}, ::Val{A}) where {N, T, A}
#     CartesianIndices(tuple_project_on_axis(celldofs(cell), Val(A)))
# end

"""
Computes the global position of a point in a cell.
"""
position(cell::Cell{N, T}, point::CartesianIndex{N}) where {N, T} = cell.local_to_global(SVector{N, T}(Tuple(point) .- 1))

"""
Computes the jacobian at the point in a cell.
"""
jacobian(cell::Cell{N, T}, point::CartesianIndex{N}) where {N, T} = Aeon.Geometry.jacobian(cell.local_to_global, SVector{N, T}(Tuple(point) .- 1))


##########################
## Interfaces ############
##########################

struct Interface{N, T, A} 
    left::Cell{N, T}
    right::Cell{N, T}

    Interface{A}(left::Cell{N, T}, right::Cell{N, T}) where {N, T, A} = new{N, T, A}(left, right)
end

struct SmoothOperators{T, O, P, R, B}
    prolong::P
    restrict::R
    boundary::B

    SmoothOperators(prolong::ProlongationOperator{T, O}, restrict::RestrictionOperator{T, O}, boundary::Operator{T, O}) where {T, O} = new{T, O, typeof(prolong), typeof(restrict), typeof(boundary)}(prolong, restrict, boundary)
end

function smooth_interface!(result::AbstractVector{T}, opers::SmoothOperators{T}, x::AbstractVector{T}, interface::Interface{N, T, A}, strength::T) where {N, T, A}
    # Compute views into functions
    leftx = cellfield(interface.left, x)
    rightx = cellfield(interface.right, x)
    leftresult = cellfield(interface.left, result)
    rightresult = cellfield(interface.right, result)

    # Compute operator tuples
    opers_identity = ntuple(dim -> dim == a ? oper.boundary : IdentityOperator(), Val(N))
    opers_prolong = ntuple(dim -> dim == a ? oper.boundary : opers.prolong, Val(N))
    opers_restrict = ntuple(dim -> dim == a ? oper.boundary : opers.restrict, Val(N))

    if interface.left.refinement == interface.right.refinement
        for index in cell_indices_on_axis(interface.left, Val(A))
            leftindex = index_splice_on_axis(index, Val(A), interface.left.dofs[A])
            rightindex = index_splice_on_axis(index, Val(A), 1)

            leftvalue = product(leftindex, opers_identity, leftx)
            rightvalue = product(rightindex, opers_identity, rightx)

            leftresult[leftindex] += strength * (leftvalue - rightvalue)
            rightresult[rightindex] += strength * (rightvalue - leftvalue)
        end
    elseif interface.left.refinement > interface.right.refinement
        for index in cell_indices_on_axis(interface.left, Val(A))
            leftindex = index_splice_on_axis(index, Val(A), interface.left.dofs[A])
            rightindex = index_splice_on_axis(index, Val(A), 1)

            leftvalue = product(leftindex, opers_identity, leftx)
            rightvalue = product(rightindex, opers_prolong, rightx)

            leftresult[leftindex] += strength * (leftvalue - rightvalue)
        end 

        for index in cell_indices_on_axis(interface.right, Val(A))
            leftindex = index_splice_on_axis(index, Val(A), interface.left.dofs[A])
            rightindex = index_splice_on_axis(index, Val(A), 1)
            
            leftvalue = product(leftindex, opers_restrict, leftx)
            rightvalue = product(rightindex, opers_identity, rightx)

            rightresult[rightindex] += strength * (rightvalue - leftvalue)
        end
    else
        for index in cell_indices_on_axis(interface.right, Val(A))
            leftindex = index_splice_on_axis(index, Val(A), interface.left.dofs[A])
            rightindex = index_splice_on_axis(index, Val(A), 1)

            leftvalue = product(leftindex, opers_prolong, leftx)
            rightvalue = product(rightindex, opers_identity, rightx)

            rightresult[rightindex] += strength * (rightvalue - leftvalue)
        end 

        for index in cell_indices_on_axis(interface.left, Val(A))
            leftindex = index_splice_on_axis(index, Val(A), interface.left.dofs[A])
            rightindex = index_splice_on_axis(index, Val(A), 1)

            leftvalue = product(leftindex, opers_identity, leftx)
            rightvalue = product(rightindex, opers_restrict, rightx)

            leftresult[leftindex] += strength * (leftvalue - rightvalue)
        end
    end
end