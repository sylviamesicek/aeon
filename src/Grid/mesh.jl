# Exports

export GridMesh, hyperrectangle, hypercube
export gridinterior, gridboundary, gridghost, gridconstraint

# Mesh

const gridinterior = 0
const gridboundary = 1
const gridghost = 2
const gridconstraint = 3

"""
Represents a grid mesh, aka a coordinate aligned grid of points whose supports fall in a regular unit square
pattern. This can either be a fully uniform mesh, or a refinement of a coarser grid mesh.
"""
struct GridMesh{N, T} 
    # The base mesh.
    mesh::Mesh{N, T}
    # A matrix of support indices.
    supports::Matrix{Int}
    # The scale of each point in the grid.
    scales::Vector{T}
    # The points in 1:prev belong to the next coarsest mesh in the refinement ladder.
    prev::Int
end

"""
    hyperrectangle(origin, dims, width, supportradius)

Builds the coarsest level of a grid mesh hierarchy. This mesh will begin at the origin, and extend `dims * width` in each
direction. Finally, there is an additional `supportradius` number of ghost points in the normal direction of each boundary.
"""
function hyperrectangle(origin::SVector{N, T}, dims::SVector{N, Int}, width::T, supportradius::Int) where {N, T}
    @assert N > 0 "Number of dimensions must be greater than 0"

    # Width of support
    supportwidth = 2supportradius + 1

    # points on each size
    pointdims = Tuple(dims .+ 1)

    # Dimensions covering full grid
    globaldims = Tuple(pointdims .+ 2supportradius)
    # Dimensions of support
    supportdims = ntuple(_ -> supportwidth, Val(N))
    # Dimensions of refined support
    refineddims = ntuple(_ -> 2supportwidth - 1, Val(N))

    pointcount = prod(globaldims)
    supportcount = prod(supportdims)

    positions = Vector{SVector{N, T}}(undef, pointcount)
    kinds = Vector{Int}(undef, pointcount)
    tags = Vector{Int}(undef, pointcount)
    supports = Matrix{T}(undef, supportcount, pointcount)

    fill!(tags, 0)

    globalcoords = CartesianIndices(globaldims)
    supportcoords = CartesianIndices(supportdims)

    globallinear = LinearIndices(globaldims)
    refinedlinear = LinearIndices(refineddims)

    for globalcoord in globalcoords
        # Some shared information
        gcoord = Tuple(globalcoord)
        gi::Int = globallinear[gcoord...]

        # Clamped point (to enable centered stencils)
        gcoord_clamped = map_tuple_with_index(gcoord) do i, x
            clamp(x, supportradius + 1, pointdims[i] + supportradius)
        end

        # Find poisition
        offsetcoord = gcoord .- supportradius .- 1
        positions[gi] = origin .+ width .* offsetcoord

        # Build support
        for (si, supportcoord) in enumerate(supportcoords)
            # Local coordinate of support point
            scoord = Tuple(supportcoord)
            # Global coordinate of support point
            sgcoord = gcoord_clamped .+ scoord .- (supportradius + 1)

            # Write into local_to_global matrix
            supports[si, gi] = globallinear[sgcoord...]
        end

        # Bool vector for whether the boundary point is an edge in that dimension
        exterior = (gcoord .< (supportradius + 1)) .|| (gcoord .> (pointdims .+ supportradius))
        boundary = (gcoord .== (supportradius + 1)) .|| (gcoord .== (pointdims .+ supportradius))
        
        # Find kind and tag
        if any(exterior)
            faces = (gcoord .≤ (supportradius + 1)) .|| (gcoord .≥ (pointdims .+ supportradius))
            # Offset vector from coord to "center"
            diag = abs.(gcoord_clamped .- gcoord)

            maxdiag = maximum(diag)

            if (diag ./ maxdiag) == faces
                # We are on ghost point
                kinds[gi] = gridghost
                tags[gi] = maxdiag
            else
                kinds[gi] = gridconstraint
                sorigin = gcoord_clamped .- supportradius
                rcoord = 2 .* (gcoord .- sorigin) .+ 1

                tags[gi] = refinedlinear[rcoord...]
            end
        elseif any(boundary)
            kinds[gi] = gridboundary
            tags[gi] = 0
        else
            kinds[gi] = gridinterior
            tags[gi] = 0
        end

    end 

    mesh = Mesh(positions, kinds, tags)

    scales = fill(width, pointcount)

    GridMesh{N, T}(mesh, supports, scales, 0)
end

function hypercube(::Val{N}, start::T, finish::T, divisions::Int, supportradius::Int) where {N, T}
    @assert finish > start

    origin = SVector(ntuple(_ -> start, Val(N)))
    dims = SVector(ntuple(_ -> divisions, Val(N)))
    width = (finish - start) / divisions

    hyperrectangle(origin, dims, width, supportradius)
end
