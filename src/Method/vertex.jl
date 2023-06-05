##########################
## Exports ###############
##########################

export Vertex, uniform_grid_vertex

##########################
## Vertex ################
##########################

"""
A collection of support points (and a primary point) in vertex space
"""
struct Vertex{D, F}
    support::Vector{SVector{D, F}}
    primary::Int
end

Base.length(v::Vertex) = length(v.support)
Base.eachindex(v::Vertex) = eachindex(v.support)
Base.getindex(v::Vertex, i) = getindex(v.support, i)

function uniform_grid_vertex(D, F, n::Int)
    coords = [LinRange(-one(F), one(F), 2n + 1) for i in 1:D]
    support = vec(collect(map(x -> SVector{D, F}(x), Iterators.product(coords...))))

    primary = ((2n + 1)^D + 1)/2
    
    Vertex{D, F}(support, primary)
end