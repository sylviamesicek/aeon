export SplitIndex, splitindices, splitreverse
export FaceIndex, faceside, faceaxis, facereverse, faceindices

"""
Index into a 2-dimensional, rank N tensor. Commonly used to index subnodes in a quadtree or along faces.
"""
struct SplitIndex{N} 
    linear::Int

    """
    Converts a linear index into a split index.
    """
    function SplitIndex{N}(linear::Int) where N
        new{N}(linear)
    end
end

Base.length(::SplitIndex{N}) where N = N
Base.getindex(split::SplitIndex{N}) where N = (UInt(split.linear - 1) & UInt(1 << (dim - 1))) > 0

SplitIndex{N}(linear::UInt) where N = SplitIndex{N}(Int(linear))

"""
Converts a tuple of booleans into a split index.
"""
function SplitIndex(cart::NTuple{N, Bool}) where N
    inner = 0x0

    for dim in 1:N
        inner |= UInt(cart[dim] << (dim - 1))
    end

    SplitIndex{N}(inner + 1)
end

function Tuple(split::SplitIndex{N}) where N 
    ntuple(dim -> split[dim], Val(N))
end

splitindices(::Val{N}) where N = Iterators.map(1:2^N) do linear
    SplitIndex{N}(linear)
end

splitreverse(split::SplitIndex{N}, axis::Int) where N = SplitIndex{N}((UInt(split.linear - 1) ⊻ UInt(1 << (axis - 1))) + 1)


struct FaceIndex{N}
    linear::Int

    FaceIndex{N}(linear::Int) where N = new{N}(linear)
end

FaceIndex{N}(axis::Int, side::Bool) where N = FaceIndex{N}(ifelse(side, axis + N, axis))

faceside(index::FaceIndex{N}) where N = index.linear > N
faceaxis(index::FaceIndex{N}) where N = ifelse(index.linear > N, index.linear - N, index.linear)
facereverse(index::FaceIndex{N}) where N = FaceIndex{N}(ifelse(index.linear > N, index.linear - N, index.linear + N))

faceindices(::Val{N}) where N = Iterators.map(1:2*N) do face
    FaceIndex{N}(face)
end


