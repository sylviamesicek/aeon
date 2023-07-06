export Block, blockcells, cellindices, cellwidths, cellcenter
export Field, evaluate, interface
export Stencil, CellStencil, VertexStencil, Coefficients

abstract type Stencil{T} end

"""
A stencil centered on a cell
"""
struct CellStencil{T, O} <: Stencil{T}
    left::SVector{O, T}
    central::T
    right::SVector{O, T}
end

stencilmaxleft(::CellStencil{T, O}, index::Int) where {T, O} = min(O, index - 1)
stencilmaxright(::CellStencil{T, O}, index::Int, total::Int) where {T, O} = min(O, total - index)
stencilsupport(::CellStencil{T, O}) where {T, O} = O

"""
A stencil centered on a vertex (but still using a cell)
"""
struct VertexStencil{T, O} <: Stencil{T}
    left::SVector{O, T}
    right::SVector{O, T}
end

stencilmaxleft(::VertexStencil{T, O}, index::Int) where {T, O} = min(O, index - 1)
stencilmaxright(::CellStencil{T, O}, index::Int, total::Int) where {T, O} = min(O, total - index + 1)
stencilsupport(::CellStencil{T, O}) where {T, O} = O

struct Coefficients{T, L, O}
    values::NTuple{L, T}

    Coefficients{O}(values::NTuple{L, T}) where {T, L, O} = new{T, L, O}(values)
end

Base.length(coefs::Coefficients) = length(coefs.values)
Base.eachindex(coefs::Coefficients) = eachindex(coefs.values)
Base.getindex(coefs::Coefficients, i::Int) = coefs.values[i]

"""
An abstract domain on which an operator may be applied
"""
abstract type Block{N, T} end

blockcells(::Block) = error("Unimplemented")

cellindices(block::Block) = CartesianIndices(blockcells(block))
cellwidths(block::Block{N, T}) where {N, T} = SVector{N, T}(1 ./ blockcells(block))
cellcenter(block::Block{N, T}, cell::CartesianIndex{N}) where {N, T} = SVector{N, T}((cell.I .- T(1//2)) ./ blockcells(block))

"""
A field defined over a domain (including the current block)
"""
abstract type Field{N, T} end

"""
Evaluates the operation of a stencil on a field in a certain block at a certain cell.
"""
function evaluate(point::CartesianIndex{N}, block::Block{N, T}, stencils::NTuple{N, Stencil{T}}, field::BlockField{N, T}) where {N, T}
    return evaluate(point, block, field, stencils...)
end

"""
Evaluates the value of a field at a cell in a block.
"""
evaluate(point::CartesianIndex{N}, block::Block{N, T}, field::Field{N, T}) where {N, T} = error("Evaluation unimplemented")

"""
Extends a centered stencil out of an interface of a block.
"""
interface(point::CartesianIndex{N}, block::Block{N, T}, field::Field{N, T}, face::FaceIndex{N}, coefficients::Coefficients{T}, rest::Stencil{T}...) where {N, T} = error("Interface evaluation unimplemented.")

"""
Recursive function for evaluating the tensor product of a stencil
"""
function evaluate(point::CartesianIndex{N}, block::Block{N, T}, field::Field{N, T}, stencil::CellStencil{T}, rest::Stencil{T}...) where {N, T}
    result = stencil.central * evaluate(point, block, field, rest...)
    result += evaluate_left(point, block, field, false, stencil.left, rest...)
    result += evaluate_right(point, block, field, false, stencil.right, rest...)
    return result
end

function evaluate(point::CartesianIndex{N}, block::Block{N, T}, field::Field{N, T}, stencil::VertexStencil{T}, rest::Stencil{T}...) where {N, T}
    result = zero(T)
    result += evaluate_left(point, block, field, true, stencil.left, rest...)
    result += evaluate_right(point, block, field, true, stencil.right, rest...)
    return result
end

function evaluate_left(point::CartesianIndex{N}, block::Block{N, T}, field::Field{N, T}, parity::Bool, stencil::NTuple{O, T}, rest::Stencil{T}...) where {N, T, O}
    axis = N - length(rest)
    index = point[axis]

    maxleft = min(O, index - 1)

    result = zero(T)

    # Iterate all left points in block
    for loff in 1:maxleft
        offsetpoint = CartesianIndex(setindex(point.I, index - loff, axis))
        result += stencil[loff] * evaluate(offsetpoint, block, field, rest...)
    end

    # If there are left points out of block, we call interface function on the remaining coefficients
    if maxleft < O
        face = FaceIndex{N}(axis, false)

        coefs = Coefficients{O}(ntuple(O - maxleft) do i
            stencil[maxleft + i]
        end)

        result += interface(point, block, field, face, coefs, rest...)
    end

    return result
end

function evaluate_right(point::CartesianIndex{N}, block::Block{N, T}, field::Field{N, T}, parity::Bool, stencil::NTuple{O, T}, rest::Stencil{T}...) where {N, T, O}
    axis = N - length(rest)
    index = point[axis]
    total = blockcells(block)[axis]

    maxright = min(O, total - index + parity)

    result = zero(T)

    # Iterate all left points in block
    for roff in 1:maxright
        offsetpoint = CartesianIndex(setindex(point.I, index - roff, axis))
        result += stencil[roff] * evaluate(offsetpoint, block, field, rest...)
    end

    # If there are left points out of block, we call interface function on the remaining coefficients
    if maxright < O
        face = FaceIndex{N}(axis, true)

        coefs = Coefficients{O}(ntuple(O - maxright) do i
            stencil[maxright + i]
        end)

        result += interface(point, block, field, face, coefs, rest...)
    end

    return result
end