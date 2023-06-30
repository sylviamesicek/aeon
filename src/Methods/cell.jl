###############
## Exports ####
###############

# export Face, cellface, boundaryface
export Cell, celldims, celltransform, cellpoints
export cellprolong!, cellrestrict!
export pointposition, pointvalue, pointgradient, pointhessian

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
    # Refinement
    refinement::Int

    function Cell(bounds::HyperBox{N, T}, refinement::Int) where {N, T}
        new{N, T}(bounds, refinement)
    end
end

"""
Returns a tuple containing the number of dofs along each dimension. This is simply a function of the refinement level of the cell.
"""
celldims(cell::Cell{N}) where {N} = ntuple(i -> 2^cell.refinement + 1, Val(N))

"""
Computes the transform of a point in cell space to global space.
"""
celltransform(cell::Cell{N}) where N = Translate(cell.bounds.origin) ∘ ScaleTransform(cell.bounds.widths)

"""
Iterates over point indices in cell.
"""
cellpoints(cell::Cell) = CartesianIndices(celldims(cell))

###########################
## Restrict/Prolong #######
###########################

function cellprolong!(cell::Cell{N, T}, refinedfunc::AbstractArray{N, T}, prolong::ProlongationOperator{T}, coarsefunc::AbstractArray{N, T}) where {N, T}
    opers = ntuple(dim -> prolong, Val(N))

    for point in cellpoints(cell)
        refinedfunc[point] = evaluate(point, opers, coarsefunc)
    end
end

function cellrestrict!(cell::Cell{N, T}, coarsefunc::AbstractArray{N, T}, restrict::RestrictionOperator{T}, refinedfunc::AbstractArray{N, T}) where {N, T}
    opers = ntuple(dim -> restrict, Val(N))

    for point in cellpoints(cell)
        coarsefunc[point] = evaluate(point, opers, refinedfunc)
    end
end

######################
## Operators #########
######################

"""
Computes the position of a point in cellspace.
"""
pointposition(cell::Cell{N, T}, point::CartesianIndex{N}) where {N, T} = SVector{N, T}((point.I .- 1) .// 2^cell.refinement)

"""
Computes the value of a func at a point.
"""
pointvalue(::Cell{N, T}, point::CartesianIndex{N}, func::AbstractArray{T, N}) where {N, T} = func[point]

"""
Computes the gradient of a func at a point in cellspace.
"""
pointgradient(cell::Cell{N, T}, point::CartesianIndex{N}, grad::Gradient{N, T}, func::AbstractArray{T, N}) where {N, T} = evaluate(point, grad, func) ./ 2^cell.refinement

"""
Computes the hessian of a func at a point in cellspace.
"""
pointhessian(cell::Cell{N, T}, point::CartesianIndex{N}, hess::Hessian{N, T}, func::AbstractArray{T, N}) where {N, T} = evaluate(point, hess, func) ./ 4^cell.refinement

##########################
## Interfaces ############
##########################

# struct Interface{N, T, A} 
#     left::Cell{N, T}
#     right::Cell{N, T}

#     Interface{A}(left::Cell{N, T}, right::Cell{N, T}) where {N, T, A} = new{N, T, A}(left, right)
# end

# struct SmoothOperators{T, O, P, R, B}
#     prolong::P
#     restrict::R
#     boundary::B

#     SmoothOperators(prolong::ProlongationOperator{T, O}, restrict::RestrictionOperator{T, O}, boundary::Operator{T, O}) where {T, O} = new{T, O, typeof(prolong), typeof(restrict), typeof(boundary)}(prolong, restrict, boundary)
# end

# function smooth_interface!(result::AbstractVector{T}, opers::SmoothOperators{T}, x::AbstractVector{T}, interface::Interface{N, T, A}, strength::T) where {N, T, A}
#     # Compute views into functions
#     leftx = cellfield(interface.left, x)
#     rightx = cellfield(interface.right, x)
#     leftresult = cellfield(interface.left, result)
#     rightresult = cellfield(interface.right, result)

#     # Compute operator tuples
#     opers_identity = ntuple(dim -> dim == a ? oper.boundary : IdentityOperator(), Val(N))
#     opers_prolong = ntuple(dim -> dim == a ? oper.boundary : opers.prolong, Val(N))
#     opers_restrict = ntuple(dim -> dim == a ? oper.boundary : opers.restrict, Val(N))

#     if interface.left.refinement == interface.right.refinement
#         for index in cell_indices_on_axis(interface.left, Val(A))
#             leftindex = index_splice_on_axis(index, Val(A), interface.left.dofs[A])
#             rightindex = index_splice_on_axis(index, Val(A), 1)

#             leftvalue = product(leftindex, opers_identity, leftx)
#             rightvalue = product(rightindex, opers_identity, rightx)

#             leftresult[leftindex] += strength * (leftvalue - rightvalue)
#             rightresult[rightindex] += strength * (rightvalue - leftvalue)
#         end
#     elseif interface.left.refinement > interface.right.refinement
#         for index in cell_indices_on_axis(interface.left, Val(A))
#             leftindex = index_splice_on_axis(index, Val(A), interface.left.dofs[A])
#             rightindex = index_splice_on_axis(index, Val(A), 1)

#             leftvalue = product(leftindex, opers_identity, leftx)
#             rightvalue = product(rightindex, opers_prolong, rightx)

#             leftresult[leftindex] += strength * (leftvalue - rightvalue)
#         end 

#         for index in cell_indices_on_axis(interface.right, Val(A))
#             leftindex = index_splice_on_axis(index, Val(A), interface.left.dofs[A])
#             rightindex = index_splice_on_axis(index, Val(A), 1)
            
#             leftvalue = product(leftindex, opers_restrict, leftx)
#             rightvalue = product(rightindex, opers_identity, rightx)

#             rightresult[rightindex] += strength * (rightvalue - leftvalue)
#         end
#     else
#         for index in cell_indices_on_axis(interface.right, Val(A))
#             leftindex = index_splice_on_axis(index, Val(A), interface.left.dofs[A])
#             rightindex = index_splice_on_axis(index, Val(A), 1)

#             leftvalue = product(leftindex, opers_prolong, leftx)
#             rightvalue = product(rightindex, opers_identity, rightx)

#             rightresult[rightindex] += strength * (rightvalue - leftvalue)
#         end 

#         for index in cell_indices_on_axis(interface.left, Val(A))
#             leftindex = index_splice_on_axis(index, Val(A), interface.left.dofs[A])
#             rightindex = index_splice_on_axis(index, Val(A), 1)

#             leftvalue = product(leftindex, opers_identity, leftx)
#             rightvalue = product(rightindex, opers_restrict, rightx)

#             leftresult[leftindex] += strength * (leftvalue - rightvalue)
#         end
#     end
# end