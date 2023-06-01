export Space

"""
`Space` defines a set of types and variables for interacting and discritizing
problem spaces. This includes tools for visiualizing functions on domains, 
generating domains using various algorithms (uniform grid, uniform fill, etc.), 
and defining functions on domains
"""
module Space
    
# Dependencies
using LinearAlgebra
using StaticArrays

####################
## Core Types ######
####################

export Point, BoundaryMeta, Domain, Kind
export pnull, domain_meta!

const pnull = typemax(UInt)

"""
Represents the `Kind` of a point in a domain.
"""
@enum Kind begin
    interior=2
    boundary=1
    ghost=3
end

struct Point{S, F}
    position::SVector{S, F}
    kind::Kind
    meta::UInt
end

struct BoundaryMeta{S, F}
    normal::SVector{S, F}
    meta::UInt
end

"""
A `Domain` is simply a set of point positions and metadata that discritize the
problem space.
"""
struct Domain{S, F}
    points::Vector{Point{S, F}}
    bounds::Vector{BoundaryMeta{S, F}}

    function Domain{S, F}(positions::Vector{SVector{S, F}}) where {S, F<:Real}
        points = map(positions) do pos
            Point{S, F}(pos, interior, 0)
        end

        new{S, F}(points, Vector{BoundaryMeta{S, F}}())
    end
end

Base.length(domain::Domain) = length(domain.points)
Base.eachindex(domain::Domain) = eachindex(domain.points)
Base.getindex(domain::Domain, i) = getindex(domain.points, i)
Base.setindex!(domain::Domain, v::Point, i) = setindex!(domain.points, v, i)
Base.firstindex(domain::Domain) = firstindex(domain.points)
Base.lastindex(domain::Domain) = lastindex(domain.points)

function domain_meta!(domain::Domain, meta::BoundaryMeta)::UInt
    index = length(domain.bounds)
    push!(domain.bounds, meta)
    index
end

# Includes
include("grid.jl")
include("writer.jl")

end