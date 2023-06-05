##########################
## Exports ###############
##########################

export GridMesh

##########################
## Grid ##################
##########################

struct GridMesh{N, T, L}
    tree::CubeTree{N, T, L, Nothing}

    function GridMesh(origin::SVector{N, T}, width::T) where {N, T}
        new{N, T, 2^N}(CubeTree(origin, width, nothing))
    end
end