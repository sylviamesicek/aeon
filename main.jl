using Aeon
using Aeon.Geometry
using Aeon.Methods
using Aeon.Operators

using LinearAlgebra
using StaticArrays

struct Poisson{N, T} <: System{N, T} end

function apply_operator!(result::AbstractVector{T}, ::Poisson{N, T}, x::AbstractVector{T}, mesh::Mesh{N, T}, ) where {N, T}
    # Clear result
    fill!(result, zero(T))

    # Using standard, 4-th order SBP coefficients.
    operator = SBPOperator{Float64, 2}(MattssonNordström2004())

    # Loop over each cell index
    for cindex in eachindex(mesh)
        # Cache the cell
        cell = mesh[cindex]
        # Loop over every point applying the laplacian
        for point in eachindex(cell)
            # Laplacian
            laplace = laplacian(cell, point, x, operator)
            # Result
            ptr = local_to_global(cell, point)
            result[ptr] += laplace
        end
        
        for dim in 1:N
            if cell.faces[dim][1] > 0
                
            end
        end
        # Enforce smoothness over interfaces 
        for face in cell.faces
            if face[1] > 0

            end
        end
    end
end

function compute_rhs!(b::AbstractVector{T}, mesh::Mesh{N, T}, system::Poisson{N, T}) where {N, T}
    fill!(b, 0.0)
end

# Main code
function main()
    mesh = hyperprism(SA[0.0, 0.0], SA[4, 4], SA[1.0, 1.0], SA[3, 3])
    
    @show mesh.doftotal

    for _ in 1:3
        refine!(mesh) do cell
            cent = center(cell.bounds)
            norm(cent) ≤ 1.0
        end
    end

    @show mesh.doftotal


    writer = MeshWriter(mesh)
    attrib!(writer, IndexAttribute())
    attrib!(writer, CellAttribute())
    write_vtk(writer, "output")

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
end

# Execute
main()