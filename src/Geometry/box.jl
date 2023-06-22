# Exports
export HyperBox, center, contains, split

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
function split(box::HyperBox, i::CartesianIndex)
    halfwidths = box.widths ./ 2

    HyperBox(box.origin .+ SVector(Tuple(i) .- 1) .* halfwidths, halfwidths)
end
