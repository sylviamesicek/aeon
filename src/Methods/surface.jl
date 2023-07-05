struct TreeSurface{N, T} 
    mesh::TreeMesh{N, T}

    total::Int
    offsets::Vector{Int}
    active::Vector{Int}
    
end