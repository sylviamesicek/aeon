using Aeon
using Aeon.Geometry
using Aeon.Methods
using Aeon.Operators

using LinearAlgebra
using LinearMaps
using StaticArrays
using IterativeSolvers

# Main code
function main()
    # # Generic Settings
    # supportradius = 2
    # basisorder = 4

    # # Build root of mesh
    # grid = hypercube(Val(2), -5.0, 5.0, 100, supportradius)

    # domain = griddomain(Val(2), Val(Float64), supportradius)
    # basis = monomials(Val(2), Val(Float64), basisorder)
    # engine = SquareEngine(domain, basis)

    # interior = filterindices(grid.mesh, gridinterior)
    # boundary = filterindices(grid.mesh, gridboundary)
    # ghost = filterindices(grid.mesh, gridghost)
    # constraint = filterindices(grid.mesh, gridconstraint)

    # pointcount = length(grid.mesh)

    # # Analytic Operators
    # curvop = ACurvature{2, Float64}()

    # # Approximations at origin
    # supportorigin = zero(SVector{2, Float64})
    # curv = approx(engine, curvop, supportorigin)

    # operator = LinearMap(pointcount) do y, x
    #     fill!(y, 0)

    #     for (_, gi) in enumerate(interior)
    #         position = grid.mesh.positions[gi]
    #         scale = grid.scales[gi]
    #         trans = ScaleTransform{2, Float64}(scale)

    #         for si in eachindex(domain)
    #             gsi = grid.supports[si, gi]
    #             stencil = transform(trans, position, curv.stencil[si])
    #             Δ = dot(stencil.inner, I)

    #             y[gi] += Δ * x[gsi]
    #         end
    #     end

    #     for (_, gi) in enumerate(boundary)
    #         # Dirichlet
    #         y[gi] = x[gi]
    #     end

    #     for (_, gi) in enumerate(constraint)
    #         # Dirichlet
    #         y[gi] = x[gi]
    #     end

    #     for (_, gi) in enumerate(ghost)
    #         # Dirichlet
    #         y[gi] = x[gi]
    #     end
    # end

    # rhs = Vector{Float64}(undef, pointcount)

    # fill!(rhs, 0)

    # for gi in eachindex(grid.mesh)
    #     position = grid.mesh.positions[gi]

    #     rhs[gi] = ℯ^(-dot(position, position)/2)
    # end
    
    # # Solve Using BiCGStab
    # sol, ch = bicgstabl(operator, rhs, 2; max_mv_products = 1000, log = true)

    # @show ch
    
    # writer = MeshWriter(grid.mesh)
    # attrib!(writer, IndexAttribute())
    # attrib!(writer, KindAttribute())
    # attrib!(writer, TagAttribute())
    # attrib!(writer, ScalarAttribute("rhs", rhs))
    # attrib!(writer, ScalarAttribute("sol", sol))
    # write_vtk(writer, "output")
end

# Execute
main()