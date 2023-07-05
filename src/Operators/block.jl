
"""
A centered stencil that extends `O` in each direction.
"""
struct Stencil{T, O} 
    left::SVector{O, T}
    central::T
    right::SVector{O, T}
end


"""
An abstract domain on which an operator may be applied
"""
abstract type Block{N, T} end

blockcells(::Block) = error("Unimplemented")

cellindices(block::Block) = CartesianIndices(blockcells(block))
cellwidths(block::Block{N, T}) where {N, T} = SVector{N, T}(1 ./ blockcells(block))
cellcenter(block::Block{N, T}, cell::CartesianIndex{N}) where {N, T} = SVector{N, T}((cell.I .- T(1//2)) ./ blockcells(block))

"""
A field defined over a block
"""
abstract type BlockField{N, T} end

"""
Evaluates the operation of a stencil on a field in a certain block at a certain cell.
"""
function evaluate(point::CartesianIndex{N}, block::Block{N, T}, stencils::NTuple{N, Stencil{T}}, field::BlockField{N, T}) where {N, T}
    return evaluate(point, block, field, stencils...)
end

"""
Evaluates the value of a field at a cell in a block.
"""
evaluate(point::CartesianIndex{N}, block::Block{N, T}, field::BlockField{N, T}) where {N, T} = error("Evaluation unimplemented")

"""
Extends a centered stencil out of an interface of a block.
"""
interface(point::CartesianIndex{N}, block::Block{N, T}, field::BlockField{N, T}, face::FaceIndex{N}, coefficients::AbstractVector{T}, rest::Stencil{T, O}...) where {N, T, O} = error("Interface evaluation unimplemented.")

"""
Recursive function for evaluating the tensor product of stencils
"""
function evaluate(point::CartesianIndex{N}, block::Block{N, T}, field::BlockField{N, T}, stencil::Stencil{T, O}, rest::Stencil{T}...) where {N, T, O}
    axis = N - length(rest)
    total = blockcells(block)[axis]
    # Evaluate at center
    result = stencil.central * evaluate(tree, node, point, rest...)

    maxleft = min(O, point[axis] - 1)

    # Iterate all left points in block
    for loff in 1:maxleft
        offsetpoint = CartesianIndex(ntuple(dim -> ifelse(dim == axis, point[axis] - loff, point[dim]), Val(N)))
        result += stencil.left[loff] * evaluate(tree, node, offsetpoint, rest...)
    end

    # If there are left points out of block, we call interface function on the remaining coefficients
    if maxleft < O
        face = FaceIndex{N}(axis, false)
        coefs = @view stencil.left[(maxleft + 1):O]
        result += interface(point, block, field, face, coefs, rest...)
    end

    maxright = min(total - point[axis], O)

    # Iterate all right points in block
    for roff in 1:maxright
        offsetpoint = CartesianIndex(ntuple(dim -> ifelse(dim == axis, point[axis] + roff, point[dim]), Val(N)))
        result += stencil.right[roff] * evaluate(tree, node, offsetpoint, rest...)
    end

    # If there are right points out of block we call interface function on the remaining coefficients
    if maxright < O
        face = FaceIndex{N}(axis, true)
        coefs = @view stencil.right[(maxright + 1):O]
        result += interface(point, block, field, face, coefs, rest...)
    end

    # Return results
    result
end