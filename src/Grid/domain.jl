export griddomain, griddomain_center, griddomain_edge, griddomain_corner

"""
    griddomain(Val(N), Val(T), supportradius)

Creates a cartesian unit domain for a gridmesh. This is a hypercube with sidelength 2, and has `2supportradius + 1`
support points on each side.
"""
function griddomain(::Val{N}, ::Val{T}, supportradius::Int) where {N, T}
    width = 2supportradius + 1
    coords = CartesianIndices(ntuple(_ -> width, Val(N)))

    positions = Vector{SVector{N, T}}(undef, length(coords))

    range = LinRange(-one(T), one(T), width)

    for (i, coord) in enumerate(coords)
        positions[i] = StaticArrays.sacollect(SVector{N, T}, range[coord[j]] for j in 1:N)
    end

    Domain(positions)
end

"""
The index of the centermost point of a grid domain.
"""
function griddomain_center(::Val{N}, supportradius::Int) where N
    width = 2radius + 1
    (width^N + 1)/2
end

"""
The index of the first middle edge point of a grid domain.
"""
function griddomain_edge(radius::Int)
    width = 2radius + 1
    (width + 1)/2
end

"""
The index of the first corner point of a grid domain.
"""
function griddomain_corner()
    0
end