export gridvertex

function gridvertex(N, T, n::Int)
    edge = 2n + 1
    indices = CartesianIndices(ntuple(i->edge, N))

    positions = Vector{SVector{N, T}}(undef, length(indices))

    range = LinRange(-one(T), one(T), edge)

    for (i, I) in enumerate(indices)
        positions[i] = SVector([range[I[j]] for j in 1:N]...)
    end

    primary = ((2n + 1)^D + 1)/2

    Vertex{N, T}(positions, primary)
end