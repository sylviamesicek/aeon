using Aeon.Analytic
using Aeon.Approx
using Aeon.Method
using Aeon.Grid
using Aeon.Tensor

using LinearAlgebra
using LinearMaps
using StaticArrays
using IterativeSolvers

# Main code
function main()
    method = GridMethod(SVector(-5.0, -5.0), 0.1, SVector(100, 100), 1)

    domain = griddomain(method)
    basis = monomials(Val(2), Val(Float64), 2)
    # weight = AGaussian{2, Float64}(1.0)
    engine = SquareEngine(domain, basis)

    level = last(method.levels)

    interiornodes = nodeindices(level.mesh, interior)
    boundarynodes = nodeindices(level.mesh, boundary)
    ghostnodes = nodeindices(level.mesh, ghost)
    constrainednodes = nodeindices(level.mesh, constraint)

    nodecount = length(level.mesh)
    supportcount = size(level.supports)[1]

    origin = zero(SVector{2, Float64})

    valueop::AIdentity{2, Float64} = AIdentity{2, Float64}()
    curvop::ACurvature{2, Float64} = ∇²(valueop)
    
    # value = approx(engine, valueop, origin)
    curv = approx(engine, curvop, origin)

    @show curv

    operator = LinearMap(nodecount) do y, x
        fill!(y, 0)

        for (_, gi) in enumerate(interiornodes)
            position = level.mesh[gi]
            scale = level.scales[gi]
            trans = ScaleTransform{2, Float64}(scale)

            for si in 1:supportcount
                gsi = level.supports[si, gi]
                stencil = transform(trans, position, curv.stencil[si])
                Δ = dot(stencil.inner, I)

                y[gi] += Δ * x[gsi]
            end
        end

        for (_, gi) in enumerate(boundarynodes)
            # Dirichlet
            y[gi] = x[gi]
        end

        for (_, gi) in enumerate(constrainednodes)
            # Dirichlet
            y[gi] = x[gi]
        end

        for (_, gi) in enumerate(ghostnodes)
            # Dirichlet
            y[gi] = x[gi]
        end
    end

    rhs = Vector{Float64}(undef, nodecount)

    fill!(rhs, 0)

    for gi in eachindex(level.mesh)
        position = level.mesh[gi]

        rhs[gi] = ℯ^(-dot(position, position)/2)
    end

    sol = zeros(Float64, nodecount)

    # Solve Using BiCGStab
    bicgstabl!(sol, operator, rhs, 2; max_mv_products = 1000)
    
    writer = MeshWriter(level.mesh)
    attrib!(writer, IndexAttribute())
    attrib!(writer, KindAttribute())
    attrib!(writer, ScalarAttribute("rhs", rhs))
    attrib!(writer, ScalarAttribute("sol", sol))
    write_vtk(writer, "output")
end

main()