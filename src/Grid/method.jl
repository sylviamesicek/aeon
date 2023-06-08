##################
## Export ########
##################

export GridLevel, GridMethod, coarsest, finest

##################
## Method ########
##################

struct GridLevel{N, T}
    # The mesh cooresponding to this level
    mesh::Mesh{N, T}
    # A matrix of support indicies. Each column cooresponds to a node in this grid level.
    supports::Matrix{Int}
    # The scale of each node.
    scales::Vector{T}
    # Nodes 1:prevlevel belong to the previous level of refinement.
    prevlevel::Int
end

struct GridMethod{N, T}
    levels::Vector{GridLevel{N, T}}

    function GridMethod(origin::SVector{N, T}, width::T, dims::SVector{N, Int}, supportradius::Int) where {N, T}
        # Increment dims to account for origin row/column
        dims .+ 1
        supportwidth = supportradius + 1

        gdims = Tuple(dims .+ 2supportradius)
        ldims = ([supportwidth for i in 1:N]...,)

        ncount = prod(gdims)
        scount = prod(ldims)

        positions = Vector{SVector{N, T}}(undef, ncount)
        kinds = Vector{NodeKind}(undef, ncount)
        supports = Matrix{T}(undef, scount, ncount)

        glinear = LinearIndices(gdims)
        llinear = LinearIndices(ldims)

        gcoords = CartesianIndices(gdims)
        lcoords = CartesianIndices(ldims)

        for ICart in gcoords
            I = SVector{N, Int}(Tuple(ICart))

            # Clamped Cartesian index
            C = SVector{N, Int}(map_tuple_with_index(Tuple(I)) do x, i
                clamp(x, supportradius + 1, dims[i] + supportradius)
            end)

            # Linear Indices
            gi::Int = glinear[I...]

            offset = SVector{N, T}(I .- supportradius .- 1)

            # Because of ghost nodes, origin is not the corner of the hyper-rectangle
            positions[gi] = SVector{N, T}(origin + offset * width)

            if any(I .< supportradius + 1) || any(I .> (dims .+ supportradius))
                if all((C .== (supportradius + 1)) .|| (C .== (dims .+ supportradius)))
                    # Is on corner
                    kinds[gi] = constraint
                else
                    kinds[gi] = ghost
                end
            elseif any(I .== (supportradius + 1)) || any(I .== (dims .+ supportradius))
                kinds[gi] = boundary
            else
                kinds[gi] = interior
            end

            for SCart in lcoords
                S = SVector{N, Int}(Tuple(SCart))
                li = llinear[S...]
                G = C + S .- (supportradius + 1)
                supports[li, gi] = glinear[G...]
            end
        end

        mesh = Mesh(positions, kinds)

        scales = Vector{T}(undef, ncount)
        fill!(scales, width)

        level = GridLevel{N, T}(mesh, supports, scales, 0)

        new{N, T}([level])
    end
end

coarsest(method::GridMethod) = first(method.levels).mesh
finest(method::GridMethod) = last(method.levels).mesh

map_tuple_with_index_helper(f, t::Tuple, index) = (@inline; (f(t[1], index), map_tuple_with_index_helper(f, Base.tail(t), index + 1)...))
map_tuple_with_index_helper(f, t::Tuple{Any,}, index) = (@inline; (f(t[1], index)))
map_tuple_with_index_helper(f, t::Tuple{}, index) = ()

map_tuple_with_index(f, t::Tuple) = map_tuple_with_index_helper(f, t, 1)