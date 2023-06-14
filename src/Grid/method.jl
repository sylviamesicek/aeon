##################
## Export ########
##################

export GridLevel, GridMethod, coarsest, finest, griddomain

##################
## Method ########
##################

struct GridLevel{N, T}
    # The mesh cooresponding to this level
    mesh::Mesh{N, T}
    # A matrix of support indicies. Each column cooresponds to a node in this grid level.
    supports::Matrix{Int}
    # A Vector of meta values for each node. This is currently used to store the type of constraint.
    meta::Vector{Int}
    # The scale of each node.
    scales::Vector{T}
    # Nodes 1:prevlevel belong to the previous level of refinement.
    prevlevel::Int
end

struct GridMethod{N, T}
    levels::Vector{GridLevel{N, T}}
    supportradius::Int

    function GridMethod(origin::SVector{N, T}, width::T, dims::SVector{N, Int}, supportradius::Int) where {N, T}
        @assert N > 0 "Number of dimensions must be greater than 0"

        # Increment dims to account for origin row/column
        dims .+ 1
        supportwidth = 2supportradius + 1

        gdims = Tuple(dims .+ 2supportradius)
        ldims = ([supportwidth for i in 1:N]...,)
        rdims = ([2supportwidth - 1 for i in 1:N]...,)

        ncount = prod(gdims)
        scount = prod(ldims)

        positions = Vector{SVector{N, T}}(undef, ncount)
        kinds = Vector{NodeKind}(undef, ncount)
        supports = Matrix{T}(undef, scount, ncount)
        meta = Vector{Int}(undef, ncount)
        fill!(meta, 0)

        glinear = LinearIndices(gdims)
        llinear = LinearIndices(ldims)
        rlinear = LinearIndices(rdims)

        gcoords = CartesianIndices(gdims)
        lcoords = CartesianIndices(ldims)

        for ICart in gcoords
            gcoord = SVector{N, Int}(Tuple(ICart))

            # Clamped Cartesian index
            gcoord_clamped = SVector{N, Int}(map_tuple_with_index(Tuple(gcoord)) do x, i
                clamp(x, supportradius + 1, dims[i] + supportradius)
            end)

            # Linear Indices
            gi::Int = glinear[gcoord...]

            offset = SVector{N, T}(gcoord .- supportradius .- 1)

            # Because of ghost nodes, origin is not the corner of the hyper-rectangle
            positions[gi] = SVector{N, T}(origin + offset * width)

            if any(gcoord .≠ gcoord_clamped)
                # This is in a ghost region
                if all((gcoord_clamped .== (supportradius + 1)) .|| (gcoord_clamped .== (dims .+ supportradius)))
                    # This is on a corner
                    off = broadcast(abs, gcoord_clamped - gcoord)
                    if all(off[1] .== off)
                        kinds[gi] = ghost
                        meta[gi] = off[1]
                    else
                        kinds[gi] = constraint

                        coordorigin = gcoord_clamped .- supportradius
                        rcoord = 2(gcoord - coordorigin) .+ 1

                        meta[gi] = rlinear[rcoord...]
                    end
                else
                    kinds[gi] = ghost
                end
            elseif any(gcoord .== (supportradius + 1)) || any(gcoord .== (dims .+ supportradius))
                kinds[gi] = boundary
            else
                kinds[gi] = interior
            end

            for LCart in lcoords
                lcoord = SVector{N, Int}(Tuple(LCart))
                li = llinear[lcoord...]
                local_to_global_coord = gcoord_clamped + lcoord .- (supportradius + 1)
                supports[li, gi] = glinear[local_to_global_coord...]
            end
        end

        mesh = Mesh(positions, kinds)

        scales = Vector{T}(undef, ncount)
        fill!(scales, width)

        level = GridLevel{N, T}(mesh, supports, meta, scales, 0)

        new{N, T}([level], supportradius)
    end
end

coarsest(method::GridMethod) = first(method.levels).mesh
finest(method::GridMethod) = last(method.levels).mesh

function griddomain(method::GridMethod{N, T}) where {N, T}
    build_griddomain(Val(N), Val(T), method.supportradius)
end

map_tuple_with_index_helper(f, t::Tuple, index) = (@inline; (f(t[1], index), map_tuple_with_index_helper(f, Base.tail(t), index + 1)...))
map_tuple_with_index_helper(f, t::Tuple{Any,}, index) = (@inline; (f(t[1], index)))
map_tuple_with_index_helper(f, t::Tuple{}, index) = ()

map_tuple_with_index(f, t::Tuple) = map_tuple_with_index_helper(f, t, 1)