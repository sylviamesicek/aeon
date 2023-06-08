##################
## Export ########
##################

export GridLevel

##################
## Method ########
##################

struct GridLevel{N, T}
    mesh::Mesh{N, T}
end

struct GridMethod{N, T, A <: ApproxEngine}
    levels::Vector{GridLevel{N, T}}
    approx::A

    function GridMethod{N, T}(origin::SVector{N, T}, width::T, dims::SVector{N, T}, radius::Int, approx::A) where {N, T, A <: ApproxEngine}
        fulldims = dims .+ (2radius + 1)

        count = prod(fulldims)

        positions = Vector{SVector{N, T}}(undef, count)
        kinds = Vector{NodeKind}(undef, count)
        centers = Vector{Int}(undef, count)

        for (i, I) in enumerate(CartesianIndices(Tuple(fulldims))) 
            offset = Tuple(I) .- (radius + 1)
            positions[i] = origin .+ offset .* width

            if any(offset .< 0) || any(offset .> dims)
                kinds[i] = NodeKind.ghost
                centers[i] = 0
            elseif any(offset .== 0) || any(offset .== dims)
                kinds[i] = NodeKind.boundary
                centers[i] = i
            else
                kinds[i] = NodeKind.interior
                centers[i] = i
            end
            
        end

        mesh = Mesh{N, T}(positions, kinds)
        level = GridLevel{N, T}(mesh)

        new{N, T, A}([level], approx)
    end
end