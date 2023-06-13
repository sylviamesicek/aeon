export build_griddomain, griddomain_center, griddomain_edge, griddomain_corner

function build_griddomain(::Val{N}, ::Val{T}, radius::Int) where {N, T}
    width = 2radius + 1
    coords = CartesianIndices(ntuple(_ -> width, Val(N)))

    positions = Vector{SVector{N, T}}(undef, length(coords))

    range = LinRange(-one(T), one(T), width)

    for (i, coord) in enumerate(coords)
        positions[i] = StaticArrays.sacollect(SVector{N, T}, range[coord[j]] for j in 1:N)
    end

    Domain(positions)
end

function griddomain_center(::Val{N}, radius::Int) where N
    width = 2radius + 1
    (width^N + 1)/2
end

function griddomain_edge(radius::Int)
    width = 2radius + 1
    (width + 1)/2
end

function griddomain_corner()
    0
end