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
Base.getindex(split::SplitIndex{N}, dim::Int) where N = (UInt(split.linear - 1) & UInt(1 << (dim - 1))) > 0

SplitIndex{N}(linear::UInt) where N = SplitIndex{N}(Int(linear))

"""
Converts a tuple of booleans into a split index.
"""
@generated function SplitIndex(cart::NTuple{N, Bool}) where N
    quote
        inner = 0x0
        Base.@nexprs $N i -> inner |= UInt(cart[i] << (i - 1))
        SplitIndex{N}(inner + 1)
    end
end

"""
Transforms a split index into a tuple of booleans. 
"""
@generated function Tuple(split::SplitIndex{N}) where N 
    :(Base.@ntuple $N dim -> split[dim])
end

"""
An iterator over every `SplitIndex` in an `N` dimensional space.
"""
splitindices(::Val{N}) where N = Iterators.map(1:2^N) do linear
    SplitIndex{N}(linear)
end

"""
Returns the `SplitIndex` on the opposite side of the given axis.
"""
splitreverse(split::SplitIndex{N}, axis::Int) where N = SplitIndex{N}((UInt(split.linear - 1) ⊻ UInt(1 << (axis - 1))) + 1)

"""
Index into the faces of a `N` dimensional hyper-prism. 
"""
struct FaceIndex{N}
    linear::Int

    FaceIndex{N}(linear::Int) where N = new{N}(linear)
end

"""
Builds an `N` dimensional `FaceIndex` from an axis and a side. 
"""
FaceIndex{N}(axis::Int, side::Bool) where N = FaceIndex{N}(ifelse(side, axis + N, axis))

"""
Returns the side this face corresponds to.
"""
faceside(index::FaceIndex{N}) where N = index.linear > N

"""
Returns the axis this face corresponds to.
"""
faceaxis(index::FaceIndex{N}) where N = ifelse(index.linear > N, index.linear - N, index.linear)

"""
Returns the index corresponding to the opposite face.
"""
facereverse(index::FaceIndex{N}) where N = FaceIndex{N}(ifelse(index.linear > N, index.linear - N, index.linear + N))

"""
An iterator over every `FaceIndex` in an `N` dimensional space.
"""
faceindices(::Val{N}) where N = Iterators.map(1:2*N) do face
    FaceIndex{N}(face)
end


