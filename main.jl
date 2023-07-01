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

            nfield[point] = gpos[1]
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

    gradient = Gradient{2}(d1)
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

        # orig = SVector(ntuple(i -> Float64(0), Val(2)))

        # @show active, trans
        # @show jacobian(trans, orig)

        for point in nodepoints(mesh, active)
            lpos = pointposition(mesh, active, point)
            j = jacobian(trans, lpos)
            # gpos = trans(lpos)

            lgrad = pointgradient(mesh, active, point, gradient, nfield)
            nresult[point] = (j * lgrad)[1]
            
            # lhess = pointhessian(mesh, active, point, hessian, nfield)
            # # ghess = j' * lhess * j

            # nresult[point] = lhess[1, 1] + lhess[2, 2] # Laplacian

            value = nfield[point]

            for face in 1:4
                # Decode face
                side = faceside(face, Val(2))
                axis = faceaxis(face, Val(2))
                edge = ifelse(side, dims[axis], 1)

                if point[axis] != edge
                    continue
                end

                neighbor = mesh.neighbors[active][face]

                if neighbor < 0
                    # Boundary conditions
                    continue
                end

                value_neighbor = smooth_interface(mesh, dofs, active, face, point, prolong, restrict, field)

                # nresult[point] += interface_strength * (value - value_neighbor)
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