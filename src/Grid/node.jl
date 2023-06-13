export grid_nodespace, grid_nodespace_center, grid_nodespace_edge, grid_nodespace_corner

function grid_nodespace(::Val{N}, ::Val{T}, radius::Int) where {N, T}
    width = 2radius + 1
    coords = CartesianIndices(ntuple(_ -> width, Val(N)))

    positions = Vector{SVector{N, T}}(undef, length(indices))

    range = LinRange(-one(T), one(T), width)

    for (i, coord) in enumerate(coords)
        positions[i] = sacollect(SVector{N, T}, range[coord[j]] for j in 1:N)
    end

    NodeSpace(positions)
end

function grid_nodespace_center(::Val{N}, radius::Int) where N
    width = 2radius + 1
    (width^N + 1)/2
end

function grid_nodespace_edge(radius::Int)
    width = 2radius + 1
    (width + 1)/2
end

function grid_nodespace_corner()
    0
end