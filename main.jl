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

    mark_refine_global!(mesh)

    prepare_and_execute_refinement!(mesh)
    
    for _ in 1:1
        for level in eachindex(mesh)
            for node in eachleafnode(mesh, level)
                bounds = nodebounds(mesh, level, node)

                if norm(bounds.origin) ≤ 2.0
                    mark_refine!(mesh, level, node)
                end
            end
        end

        prepare_and_execute_refinement!(mesh)
    end

    dofs = DoFHandler(mesh)

    @show mesh

    total = dofstotal(dofs)

    seed = Vector{Float64}(undef, total)

    for level in eachindex(mesh)
        for node in eachleafnode(mesh, level)
            transform = nodetransform(mesh, level, node)
            offset = nodeoffset(dofs, level, node)

            for (i, cell) in enumerate(cellindices(mesh))
                lpos = cellposition(mesh, cell)
                gpos = transform(lpos)

                seed[offset + i] = gunlach_laplacian(gpos, 1.0, 1.0)
            end
        end
    end

    writer = MeshWriter{2, Float64}()
    attrib!(writer, BlockAttribute())
    attrib!(writer, ScalarAttribute("seed", seed))
    write_vtu(writer, mesh, dofs, "output")
end

# Execute
main()