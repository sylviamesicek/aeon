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
export pnull, domain_tag!, domain_boundary!

const pnull = typemax(UInt)

"""
Represents the `Kind` of a point in a domain.
"""
@enum Kind begin
    interior=1
    boundary=2
    ghost=3
end

struct PointMeta
    kind::Kind
    meta::Int
end

struct BoundaryMeta{S, F}
    normal::SVector{S, F}
    meta::Int
    prev::Int
end

"""
A `Domain` is simply a set of point positions and metadata that discritize the
problem space.
"""
struct Domain{S, F}
    positions::Vector{SVector{S, F}}
    points::Vector{PointMeta}
    bounds::Vector{BoundaryMeta{S, F}}

    function Domain(positions::Vector{SVector{S, F}}) where {S, F}
        count = length(positions)
        points = Vector{PointMeta}(undef, count)

        for i in eachindex(points)
            points[i] = PointMeta(interior, 0)
        end

        new{S, F}(positions, points, Vector{BoundaryMeta{S, F}}())
    end
end

Base.length(domain::Domain) = length(domain.positions)
Base.eachindex(domain::Domain) = eachindex(domain.positions)
Base.getindex(domain::Domain, i) = getindex(domain.positions, i)
Base.setindex!(domain::Domain, v::SVector, i) = setindex!(domain.positions, v, i)
Base.firstindex(domain::Domain) = firstindex(domain.positions)
Base.lastindex(domain::Domain) = lastindex(domain.positions)

"""
    domain_tag!(domain, pidx, tag)

Associates a point in the domain with a tag. This is simply an `Int` associated with the point used for differentiating
the parts of the domain.
"""
function domain_tag!(domain::Domain, pidx::Int, tag::Int)
    if domain.points[pidx].kind == boundary
        bidx = domain.points[pidx].meta
        boundary = domain.bounds[bidx]
        boundary = BoundaryMeta(boundary.normal, tag, boundary.prev)
        domain.bounds[bidx] = boundary
    else
        point = domain[pidx]
        point = PointMeta(point.kind, tag)
        domain[pidx] = point
    end
end

"""
    domain_boundary!(domain, pidx, normal)

Transforms a point in the domain into a boundary point. This copies the meta data already associated with the point
and adds a normal vector to the point.
"""
function domain_boundary!(domain::Domain, pidx::Int, normal::SVector)
    # TODO, reuse already allocated arrays (i.e., make this function idimpotent)
    tag = domain.points[pidx].meta

    bidx = length(domain.bounds)
    push!(domain.bounds, BoundaryMeta(normal, tag, pidx))

    domain.points[pidx] = PointMeta(boundary, bidx)
end

# Includes
include("grid.jl")
include("writer.jl")

end