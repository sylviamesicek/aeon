# Exports
export Grid, mkdomain

struct Grid{Dim, Field}
    min_bound::SVector{Dim, Field}
    max_bound::SVector{Dim, Field}
    cells::SVector{Dim, UInt}

    function Grid(min::SVector{Dim, Field}, max::SVector{Dim, Field}, cells::SVector{Dim, UInt}) where {Dim, Field}
        if all(min .< max)
            return new{Dim, Field}(min, max, cells)
        end
        error("Invalid grid bounds: $(min) ≮ $(max)")
    end
end

function bdirection(min::SVector{D, F}, max:: SVector{D, F}, x::SVector{D, F})::SVector{D, F} where {D, F}
    left = map(b -> b ? -one(F) : zero(F), x .≤ min)
    right = map(b -> b ? one(F) : zero(F), max .≤ x)

    left .+ right
end

function Domain(grid::Grid{D, F}) where {D, F <: Real}
    coords = [LinRange(grid.min_bound[i], grid.max_bound[i], grid.cells[i]) for i in 1:D]
    positions = vec(collect(map(x -> SVector{D, F}(x), Iterators.product(coords...))))
    domain = Domain(positions)

    for i in eachindex(domain)
        dir = bdirection(grid.min_bound, grid.max_bound, domain[i])
        if !iszero(dir)
            domain_boundary!(domain, i, normalize(dir))
        end
    end

    domain
end