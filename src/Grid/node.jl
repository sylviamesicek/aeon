export grid_nodespace, grid_nodespace_center, grid_nodespace_edge, grid_nodespace_corner

function grid_nodespace(N, T, radius::Int)
    width = 2radius + 1
    indices = CartesianIndices(ntuple(i->width, N))

    positions = Vector{SVector{N, T}}(undef, length(indices))

    range = LinRange(-one(T), one(T), width)

    for (i, I) in enumerate(indices)
        positions[i] = SVector([range[I[j]] for j in 1:N]...)
    end

    NodeSpace(positions)
end

function grid_nodespace_center(N, radius::Int)
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