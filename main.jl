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

    mesh = Mesh(HyperBox(SA[0.0, 0.0], SA{Float64}[π, π]), 9)

    # mark_refine_global!(mesh)

    # prepare_and_execute_refinement!(mesh)
    
    # for _ in 1:1
    #     for level in eachindex(mesh)
    #         for node in eachleafnode(mesh, level)
    #             bounds = nodebounds(mesh, level, node)

    #             if norm(bounds.origin) ≤ 2.0
    #                 mark_refine!(mesh, level, node)
    #             end
    #         end
    #     end

    #     prepare_and_execute_refinement!(mesh)
    # end

    dofs = DoFHandler(mesh)
    total = dofstotal(dofs)

    @show mesh
    @show total

    seed = Vector{Float64}(undef, total)

    for level in eachindex(mesh)
        for node in eachleafnode(mesh, level)
            transform = nodetransform(mesh, level, node)
            offset = nodeoffset(dofs, level, node)

            for (i, cell) in enumerate(cellindices(mesh))
                lpos = cellposition(mesh, cell)
                gpos = transform(lpos)

                # seed[offset + i] = gunlach_laplacian(gpos, 1.0, 1.0)
                seed[offset + i] = sin(gpos.x)*sin(gpos.y)
                # seed[offset + i] = 1.0
            end
        end
    end

    block = Block{Float64, 2}(undef, nodecells(mesh)...)

    hemholtz = LinearMap(total) do y, x
        for level in eachindex(mesh)
            for node in eachleafnode(mesh, level)
                offset = nodeoffset(dofs, level, node)
                transform = nodetransform(mesh, level, node)
                
                # Fill Interior
                fill_interior_from_linear!(block) do i
                    x[offset + i]
                end

                # Apply boundary conditions
                block_boundary(block) do boundary
                    # lpos = cellposition(block, cell)
                    # gpos = transform(lpos)
                    # j = inv(jacobian(transform, lpos))

                    # value = 0.0
                    # coefficient = 0.0

                    # gradient = SVector(ntuple(Val(2)) do i
                        
                    # end)

                    block_diritchlet!(block, boundary, basis, 1.0, 0.0)
                end

                for (i, cell) in enumerate(cellindices(block))
                    lpos = cellposition(block, cell)
                    j = inv(jacobian(transform, lpos))

                    # gpos = transform(lpos)

                    lhess = blockhessian(block, cell, basis)
                    ghess = j' * lhess * j

                    y[offset + i] = -(ghess[1, 1] + ghess[2, 2])
                end
            end
        end
    end

    solution, history = bicgstabl(hemholtz, seed, 2; log=true, max_mv_products=8000)

    @show history

    # solution = hemholtz * seed

    writer = MeshWriter{2, Float64}()
    attrib!(writer, NodeAttribute())
    attrib!(writer, ScalarAttribute("seed", seed))
    attrib!(writer, ScalarAttribute("solution", solution))
    write_vtu(writer, mesh, dofs, "output")
end

# Execute
main()