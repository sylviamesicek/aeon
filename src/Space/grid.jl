# Exports
export Grid, mkdomain

struct Grid{S, F}
    min_bound::SVector{S, F}
    max_bound::SVector{S, F}
    cells::SVector{S, UInt}

    function Grid{S, F}(min::SVector{S, F}, max::SVector{S, F}, cells::SVector{S, UInt}) where {S, F}
        if all(min .< max)
            return new{S,F}(min, max, cells)
        end
        error("Invalid grid bounds: $(min) ≮ $(max)")
    end
end

function bdirection(min::SVector{S, F}, max:: SVector{S, F}, x::SVector{S, F})::SVector{S, F} where {S, F}
    left = map(b -> b ? -one(F) : zero(F), x .≤ min)
    right = map(b -> b ? one(F) : zero(F), max .≤ x)

    left .+ right
end

function mkdomain(grid::Grid{S, F})::Domain{S, F} where {S, F <: Real}
    coords = [LinRange(grid.min_bound[i], grid.max_bound[i], grid.cells[i]) for i in 1:S]
    positions = vec(collect(map(x -> SVector{S, F}(x), Iterators.product(coords...))))
    domain = Domain{S, F}(positions)

    for i in eachindex(domain)
        dir = bdirection(grid.min_bound, grid.max_bound, domain[i].position)
        if !iszero(dir)
            position = copy(domain[i].position)
            meta = domain_meta!(domain, BoundaryMeta(normalize(dir), zero(UInt)))

            domain[i] = Point(position, boundary, meta)
        end
    end

    domain
end