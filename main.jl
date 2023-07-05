using Aeon
using Aeon.Geometry
using Aeon.Methods
using Aeon.Operators

using LinearAlgebra
using StaticArrays
using LinearMaps
using IterativeSolvers

function apply_operator!(y::AbstractVector, mesh::TreeMesh, dofs::DoFManager, x::AbstractVector)
    ###############################
    ## Build numerical operators ##
    ###############################

    source = MattssonNordström2004{Float64, 2}()

    d1 = derivative_operator(source, Val(1))
    d2 = derivative_operator(source, Val(2))

    prolong = prolongation_operator(source)
    restrict = restriction_operator(source)

    boundary = boundary_derivative_operator(source, Val(1))

    gradient = Gradient{2}(d1)
    hessian = Hessian{2}(d1, d2)

    ###############################
    ## Apply operator to field ####
    ###############################

    interface_strength = -1
    boundary_strength = -1

    for active in dofs.active
        trans = nodetransform(mesh, active)
        dims = nodedims(mesh, active)

        nx = nodefield(mesh, dofs, active, x)
        ny = nodefield(mesh, dofs, active, y)

        for point in nodepoints(mesh, active)
            lpos = pointposition(mesh, active, point)
            j = inv(jacobian(trans, lpos))
            
            lhess = pointhessian(mesh, active, point, hessian, nx)
            ghess = j' * lhess * j

            ny[point] = ghess[1, 1] + ghess[2, 2]

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
                    ny[point] += boundary_strength * nx[point]
                else
                    val = smooth_interface(mesh, dofs, active, face, point, prolong, restrict, IdentityOperator{Float64, 2}(), x)
                    # der = smooth_interface(mesh, dofs, active, face, point, prolong, restrict, boundary, x)
                    ny[point] += interface_strength * val
                end
            end
        end
    end
end

# Main code
function main()
    ##########################
    ## Build mesh ############
    ##########################

    mesh = TreeMesh(HyperBox(SA[0.0, 0.0], SVector{2, Float64}(π, π)), 5)

    for _ in 1:0
        refine!(mesh) do node, _
            # norm(mesh.bounds[node].origin) < 0.1
            true
        end
    end

    ###########################
    ## Build refined level ####
    ###########################

    dofs = DoFManager(mesh)

    @show dofs.total
    @show dofs.active

    # Build field
    rhs = Vector{Float64}(undef, dofs.total)

    for active in dofs.active
        trans = nodetransform(mesh, active)
        nrhs = nodefield(mesh, dofs, active, rhs)

        for point in nodepoints(mesh, active)
            lpos = pointposition(mesh, active, point)
            gpos = trans(lpos)

            # nfield[point] = cos(gpos.x) + cos(gpos.y)
            nrhs[point] = -2sin(gpos.x)*sin(gpos.y)
        end
    end

    sol_analytic = Vector{Float64}(undef, dofs.total)

    for active in dofs.active
        trans = nodetransform(mesh, active)
        nanalytic = nodefield(mesh, dofs, active, sol_analytic)

        for point in nodepoints(mesh, active)
            lpos = pointposition(mesh, active, point)
            gpos = trans(lpos)

            nanalytic[point] = sin(gpos.x)*sin(gpos.y)
        end
    end

    operator = LinearMap(dofs.total) do y, x
        apply_operator!(y, mesh, dofs, x)
    end

    # result = operator * field
    # diff = result .- analytical

    # sol_numeric = operator * sol_analytic

    # @show size(operator, 2)

    # Solve Using BiCGStab
    sol_numeric, history = bicgstabl(operator, rhs, 2; log = true)

    @show history

    sol_diff = sol_numeric .- sol_analytic

    writer = MeshWriter(mesh)
    attrib!(writer, IndexAttribute())
    attrib!(writer, CellAttribute())
    attrib!(writer, ScalarAttribute{2}("rhs", rhs))
    attrib!(writer, ScalarAttribute{2}("sol_numeric", sol_numeric))
    attrib!(writer, ScalarAttribute{2}("sol_analytic", sol_analytic))
    attrib!(writer, ScalarAttribute{2}("sol_diff", sol_diff))
    write_vtu(writer, dofs, "output")

    values = lagrange([-1.0, 0.0, 1.0], 0.0)
    @show values
end

# Execute
main()