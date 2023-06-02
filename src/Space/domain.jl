
export PointMeta, BoundaryMeta, Domain, Kind
export pnull, position_array, normal_array, domain_tag!, domain_boundary!

const pnull::UInt = typemax(UInt)

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

struct BoundaryMeta
    prev::Int
    meta::Int
end

"""
A `Domain` is simply a set of point positions and metadata that discritize the
problem space.
"""
struct Domain{Dim, Field}
    # Data Vectors
    positions::Vector{SVector{Dim, Field}}
    normals::Vector{SVector{Dim, Field}}

    # Meta Vectors
    points::Vector{PointMeta}
    bounds::Vector{BoundaryMeta}

    function Domain(positions::Vector{SVector{Dim, Field}}) where {Dim, Field}
        points = Vector{PointMeta}(undef, length(positions))
    
        for i in eachindex(points)
            points[i] = PointMeta(interior, 0)
        end
    
        new{Dim, Field}(positions, Vector{SVector{Dim, Field}}(), points, Vector{BoundaryMeta}())
    end
end



Base.length(domain::Domain) = length(domain.positions)
Base.eachindex(domain::Domain) = eachindex(domain.positions)
Base.getindex(domain::Domain, i) = getindex(domain.positions, i)
Base.setindex!(domain::Domain, v::SVector, i) = setindex!(domain.positions, v, i)
Base.firstindex(domain::Domain) = firstindex(domain.positions)
Base.lastindex(domain::Domain) = lastindex(domain.positions)

function position_array(domain::Domain{D, F}) where {D, F}
    array = Array{F, 2}(undef, D, length(domain))

    for i in eachindex(domain)
        for j in 1:D
            array[j, i] = domain[i][j]
        end
    end

    array
end

function normal_array(domain::Domain{D, F}) where {D, F}
    array::Array{F, 2} = zeros(D, length(domain))

    for i in eachindex(domain.bounds)
        prev = domain.bounds[i].prev

        for j in 1:D
            array[j, prev] = domain.normals[i][j]
        end
    end

    array
end


"""
    domain_tag!(domain, pidx, tag)

Associates a point in the domain with a tag. This is simply an `Int` associated with the point used for differentiating
the parts of the domain.
"""
function domain_tag!(domain::Domain, pidx::Int, tag::Int)
    if domain.points[pidx].kind == boundary
        bidx = domain.points[pidx].meta
        domain.bounds[bidx] = BoundaryMeta(pidx, tag)
    else
        point = PointMeta(domain[pidx].kind, tag)
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
    push!(domain.normals, normal)
    push!(domain.bounds, BoundaryMeta(pidx, tag))

    domain.points[pidx] = PointMeta(boundary, bidx)
end
