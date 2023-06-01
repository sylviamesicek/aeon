struct Grid{S, F<:Real}
    min_bound::SVector{S, F}
    max_bound::SVector{S, F}
    cells::SVector{S, UInt}

    # Grid(min_bound, max_bound, cells) = new{S, F}(min_bound, max_bound, cells)
end

function mkdomain(grid::Grid{S, F})::Domain{S, F} where {S, F <: Real}
    coords = [LinRange(grid.min_bound[i], grid.max_bound[i], grid.cells[i] + 2) for i in 1:S]
    grid = Iterators.product(coords...)
    positions = vec(collect(map(x -> SVector{S, F}(x), grid)))

    pointcount = length(positions)

    tags = zeros(pointcount)

    Domain{S, F}(positions, Vector{Kind}(), Vector{UInt}(), Vector{SVector{S, F}}())
end