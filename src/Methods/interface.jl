export smooth_interface

function smooth_interface(mesh::TreeMesh{N}, dofs::DoFManager{N, T}, node::Int, face::Int, point::CartesianIndex{N}, 
    prolong::ProlongationOperator{T, O}, restrict::RestrictionOperator{T, O}, boundary::Operator{T, O}, func::AbstractVector{T}) where {N, T, O}

    dims = nodedims(mesh, node)

    # Decode face
    side = faceside(face, Val(N))
    axis = faceaxis(face, Val(N))
    # edge = ifelse(side, dims[axis], 1)

    nfield = nodefield(mesh, dofs, node, func)

    opers_identity = ntuple(dim -> dim == axis ? boundary : IdentityOperator{T, O}(), Val(N))

    value = evaluate(point, opers_identity, nfield)

    # Get neighbor
    neighbor = mesh.neighbors[node][face]

    if neighbor > 0 && mesh.children[neighbor] == 0
        # Identity
        dims_neighbor = nodedims(mesh, neighbor)
        edge_neighbor = ifelse(side, 1, dims_neighbor[axis])
        point_neighbor = CartesianIndex(ntuple(dim -> ifelse(dim == axis, edge_neighbor, point[dim]), Val(N)))
        nfield_neighbor = nodefield(mesh, dofs, neighbor, func)

        return value - evaluate(point_neighbor, opers_identity, nfield_neighbor)
    elseif neighbor > 0 

        @show "Restriction"
        # Restrict
        opers_restrict = ntuple(dim -> dim == axis ? boundary : restrict, Val(N))

        value_neighbor = T(0)
        number_faces = 0

        for linear in 1:2^N
            cart = split_linear_to_cart(linear, Val(N))

            # Face does not touch child
            if cart[axis] == side
                continue
            end

            child = mesh.children[neighbor] + linear
    
            dims_child = nodedims(mesh, child)
            edge_child = ifelse(side, 1, dims_child[axis])

            point_child_refined = coarse_to_refined.(point.I) .- cart .* (dims_child .- 1)

            point_touches_child = true

            for dim in 1:N
                point_touches_child &= axis == dim || (point_child_refined[dim] ≥ 1 && point_child_refined[dim] ≤ dims_child[dim])
            end

            # Point does not touch child
            if !point_touches_child
                continue
            end

            nfield_child = nodefield(mesh, dofs, child, func)
            point_child = CartesianIndex(ntuple(dim -> ifelse(dim == axis, edge_child, refined_to_coarse(point_child_refined[dim])), Val(N)))

            value_neighbor += evaluate(point_child, opers_restrict, nfield_child)
            number_faces += 1
        end

        return value - value_neighbor/number_faces
    else
        @show "Prolongation"
        # Prolongation
        opers_prolong = ntuple(dim -> dim == axis ? boundary : prolong, Val(N))

        parent = mesh.parents[node]
        neighbor = mesh.neighbors[parent][face]
    
        dims_neighbor = nodedims(mesh, neighbor)
        edge_neighbor = ifelse(side, 1, dims_neighbor[axis])
    
        linear = node - mesh.children[parent]
        cart = split_linear_to_cart(linear, Val(N))
    
        point_neighbor = CartesianIndex(ntuple(dim -> ifelse(dim == axis, edge_neighbor, point[dim] + cart[dim] * (dims[dim] - 1)), Val(N)))
        nfield_neighbor = nodefield(mesh, dofs, neighbor, func)

        return value - evaluate(point_neighbor, opers_prolong, nfield_neighbor)
    end
end