# Exports
export HyperBox, center, contains, splitbox
export HyperFaces, nfaces

# Code

"""
An N-dimensional axis aligned bounding box.
"""
struct HyperBox{N, T}
    origin::SVector{N, T}
    widths::SVector{N, T}

    function HyperBox(origin::SVector{N, T}, widths::SVector{N, T}) where {N, T}
        new{N, T}(origin, widths)
    end
end

Base.show(io::IO, box::HyperBox) = print(io, "HyperBox($(box.origin), $(box.widths))")

"""
Computes the center point of a `HyperBox`.
"""
center(box::HyperBox) = box.origin .+ box.widths ./ 2

"""
Checks if a point is contained within a `HyperBox`.
"""
contains(box::HyperBox{N, T}, x::SVector{N, T}) where {N, T} = all(box.origin .≤ x) && all(x .≤ (box.origin .+ box.widths))

"""
Computes the subbox built by subdividing a hyperbox in half in each dimension.
"""
function splitbox(box::HyperBox{N}, index::SplitIndex{N}) where N
    halfwidths = box.widths ./ 2
    HyperBox(box.origin .+ SVector(Tuple(index)) .* halfwidths, halfwidths)
end

"""
Stores data for each face of a hyperbox.
"""
struct HyperFaces{N, T, F}
    inner::NTuple{F, T}

    function HyperFaces{N}(faces::NTuple{F, T}) where {N, T, F}
        @assert 2*N == F
        new{N, T, F}(faces)
    end
end

HyperFaces(faces::NTuple{F, T}) where {F, T} = HyperFaces{F ÷ 2}(faces)
HyperFaces(faces::T...) where T = HyperFaces(tuple(faces...))

Base.length(faces::HyperFaces) = length(faces.inner)
Base.eachindex(::HyperFaces{N}) where N = faceindices(Val(N))
Base.getindex(faces::HyperFaces{N}, index::FaceIndex{N}) where {N} = faces.inner[index.linear]
Base.show(io::IO, faces::HyperFaces) = print(io, "HyperFaces$(tuple(faces.inner...))")

function StaticArrays.setindex(faces::HyperFaces{N, T, F}, val::T, index::FaceIndex{N}) where {N, T, F}
    HyperFaces(ntuple(face -> ifelse(face == index.linear, val, faces.inner[face]), Val(F)))
end

"""
Constructs an `N` dimensional hyperfaces object from a function (analgous to the `ntuple` function for tuples).
"""
nfaces(f::Function, ::Val{N}) where N = HyperFaces(ntuple(face -> f(FaceIndex{N}(face)), Val(2*N)))