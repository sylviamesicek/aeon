using Aeon
using Aeon.Geometry
using Aeon.Methods
using Aeon.Operators

using LinearAlgebra
using StaticArrays

# Main code
function main()
    ##########################
    ## Build mesh ############
    ##########################

    mesh = TreeMesh(HyperBox(SA[0.0, 0.0], SA[1.0, 1.0]), 5)

    for _ in 1:3
        refine!(mesh) do node, _
            norm(mesh.bounds[node].origin) < 0.1
        end
    end

    ###########################
    ## Build refined level ####
    ###########################

    dofs = DoFManager(mesh)

    @show dofs.total

    # Build field
    field = Vector{Float64}(undef, dofs.total)

    for active in dofs.active
        trans = nodetransform(mesh, active)
        nfield = nodefield(mesh, dofs, active, field)

        for point in nodepoints(mesh, active)
            lpos = pointposition(mesh, active, point)
            gpos = trans(lpos)

            nfield[point] = sum(gpos .^ 2)
        end
    end

    ###############################
    ## Build numerical operators ##
    ###############################

    source = MattssonNordström2004{Float64, 2}()

    d1 = derivative_operator(source, Val(1))
    d2 = derivative_operator(source, Val(2))

    prolong = prolongation_operator(source)
    restrict = restriction_operator(source)

    hessian = Hessian{2}(d1, d2)

    ###############################
    ## Apply operator to field ####
    ###############################
    # TEMP

    interface_strength = -1
    boundary_strength = 0

    result = Vector{Float64}(undef, dofs.total)

    for active in dofs.active
        trans = nodetransform(mesh, active)
        dims = nodedims(mesh, active)

        nfield = nodefield(mesh, dofs, active, field)
        nresult = nodefield(mesh, dofs, active, result)

        for point in nodepoints(mesh, active)
            lpos = pointposition(mesh, active, point)
            j = jacobian(trans, lpos)
            # gpos = trans(lpos)
            
            lhess = pointhessian(mesh, active, point, hessian, nfield)
            ghess = j' * lhess * j

            nresult[point] = ghess[1, 1] + ghess[2, 2] # Laplacian

            value = nfield[point]

            for face in 1:4
                # Decode face
                side = faceside(face, Val(2))
                axis = faceaxis(face, Val(2))
                edge = ifelse(side, dims[axis], 1)

                # Only continue if point is along face
                if point[axis] != edge
                    continue
                end

                # Get neighbor
                neighbor = mesh.neighbors[active][face]

                # Enforce boundary conditions
                if neighbor < 0
                    nresult[point] += boundary_strength * (value - 0)

                    continue
                end

                edge_neighbor = ifelse(side, 1, dims[axis])

                # Enforce interface conditions
                if neighbor > 0 && mesh.children[neighbor] == 0
                    # Neighbor is on same level
                    nfield_neighbor = nodefield(mesh, dofs, neighbor, field)
                    point_neighbor = CartesianIndex(ntuple(dim -> ifelse(dim == axis, edge_neighbor, point[dim]), Val(2)))
                    value_neighbor = nfield_neighbor[point_neighbor]
                elseif neighbor > 0 
                    # Neighbor is more refined, restrict it to find boundary values
                    restrict_opers = ntuple(dim -> dim == axis ? IdentityOperator{Float64, 2}() : restrict, Val(2))

                    value_neighbor = 0
                    
                    for linear in 1:4
                        cart = split_linear_to_cart(linear, Val(2))

                        point_offset = cart .* (refined_to_coarse.(dims) .- 1)
                        point_child = CartesianIndex(ntuple(dim -> ifelse(dim == axis, edge_neighbor, point[dim] - point_offset[dim]), Val(2)))

                        child_shares_face = cart[axis] == side

                        for dim in 1:2
                            child_shares_face &= dim == axis || point_child[dim] > 0
                            child_shares_face &= dim == axis || point_child[dim] ≤ refined_to_coarse(dims[dim])
                        end

                        # Check that child is touching the current point
                        if !child_shares_face
                            continue
                        end

                        child = mesh.children[neighbor] + linear
                        nfield_child = nodefield(mesh, dofs, child, field)

                        value_neighbor += evaluate(point_child, restrict_opers, nfield_child)
                    end
                else
                    prolong_opers = ntuple(dim -> dim == axis ? IdentityOperator{Float64, 2}() : prolong, Val(2))

                    # Neighbor is less refined, prolong it to find boundary values
                    parent = mesh.parents[active]

                    neighbor = mesh.neighbors[parent][face]
                    nfield_neighbor = nodefield(mesh, dofs, neighbor, field)

                    linear = active - mesh.children[parent]
                    cart = split_linear_to_cart(linear, Val(2))

                    point_neighbor = CartesianIndex(ntuple(dim -> ifelse(dim == axis, edge_neighbor, point[dim] + cart[dim] * (dims[dim] - 1)), Val(2)))

                    value_neighbor = evaluate(point_neighbor, prolong_opers, nfield_neighbor)
                end

                nresult[point] += interface_strength * (value - value_neighbor)
            end
        end
    end

    writer = MeshWriter(mesh)
    attrib!(writer, IndexAttribute())
    attrib!(writer, CellAttribute())
    attrib!(writer, ScalarAttribute{2}("field", field))
    attrib!(writer, ScalarAttribute{2}("result", result))
    write_vtk(writer, dofs, "output")
end

# Execute
main()