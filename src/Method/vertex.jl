##########################
## Exports ###############
##########################

export Vertex, center

##########################
## Vertex ################
##########################
"""
A collection of support points (and a primary point) in vertex space
"""
struct Vertex{N, T}
    support::Vector{SVector{N, T}}
    primary::Int
end

Base.length(v::Vertex) = length(v.support)
Base.eachindex(v::Vertex) = eachindex(v.support)
Base.getindex(v::Vertex, i) = getindex(v.support, i)

center(vertex::Vertex) = vertex[vertex.primary]

# function uniform_grid_vertex(D, F, n::Int)
#     coords = [LinRange(-one(F), one(F), 2n + 1) for i in 1:D]
#     support = vec(collect(map(x -> SVector{D, F}(x), Iterators.product(coords...))))

#     primary = ((2n + 1)^D + 1)/2
    
#     Vertex{D, F}(support, primary)
# end