using Aeon
using Aeon.Geometry
using Aeon.Blocks
using Aeon.Methods

using IterativeSolvers
using LinearMaps
using LinearAlgebra
using StaticArrays

function gunlach_seed(pos::SVector{2, T}, A::T, σ::T) where {T}
    ρ = pos[1]
    z = pos[2]

    A * (ρ/σ)^2 * ℯ^(-(ρ^2 + z^2) / σ^2)
end

function gunlach_laplacian(pos::SVector{2, T}, A::T, σ::T) where T
    ρ = pos[1]
    z = pos[2]

    2A*(2ρ^4 - 6ρ^2*σ^2 + σ^4 + 2ρ^2 * z^2) * ℯ^(-(ρ^2 + z^2) / σ^2) / σ^6
end

# Main code
function main()
    # Function basis
    basis = LagrangeBasis{Float64}()

    # Mesh

    mesh = Mesh(HyperBox(SA[0.0, 0.0], SA{Float64}[4.0, 4.0]), 6)

    for _ in 1:3
        for level in eachindex(mesh)
            for node in eachleafnode(mesh, level)
                bounds = nodebounds(mesh, level, node)

                if norm(bounds.origin) < 2.0
                    mark_refine!(mesh, level, node)
                end
            end
        end

        prepare_and_execute_refinement!(mesh)
    end

    dofs = DoFHandler(mesh)

    @show mesh

    writer = MeshWriter{2, Float64}()
    attrib!(writer, BlockAttribute())
    write_vtu(writer, mesh, dofs, "output")
end

# Execute
main()