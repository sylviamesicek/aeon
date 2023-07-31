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

function scalar_field(pos::SVector{2, T}, A::T, σ::T, mass::T) where T
    ρ = pos[1]
    z = pos[2]


    r² = ρ^2 + z^2
    ϕ = A * ℯ^(-r² / σ^2)
    
    mass^2 * (ϕ^2 + ϕ) / 2
end

initial_data(pos::SVector{2, T}, Aₕ::T, σₕ::T, Aₛ::T, σₛ::T, mass::T) where T = (gunlach_laplacian(pos, Aₕ, σₕ) + scalar_field(pos, Aₛ, σₛ, mass)) / 4

# Main code
function main()
    # Function basis
    basis = LagrangeBasis{Float64}()

    # Mesh

    mesh = Mesh(HyperBox(SA[0.0, 0.0], SA[6.0, 6.0]), 9, 0)

    # for i in 1:2
    #     mark_refine_global!(mesh)
    #     prepare_and_execute_refinement!(mesh)
    # end

    # for _ in 1:3
    #     foreachleafnode(mesh) do level, node
    #         origin = nodebounds(mesh, level, node).origin

    #         if norm(origin) < 1.0
    #             mark_refine!(mesh, level, node)
    #         end
    #     end

    #     prepare_and_execute_refinement!(mesh)
    # end

    # mark_refine_global!(mesh)
    # prepare_and_execute_refinement!(mesh)

    # mark_refine_global!(mesh)
    # prepare_and_execute_refinement!(mesh)

    # mark_refine!(mesh, 2, 1)
    # prepare_and_execute_refinement!(mesh)

    # mark_refine_global!(mesh)
    # prepare_and_execute_refinement!(mesh)

    # mark_refine!(mesh, 4, 1)
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

    dofs = DoFManager(mesh)
    blocks = BlockManager{2}(mesh)
    total = dofstotal(dofs)

    @show mesh
    @show total

    seed = project(mesh, dofs) do pos
        initial_data(pos, 1.0, 1.0, 0.0, 0.0, 0.0)
    end

    mv_product = 1

    hemholtz = LinearMap(total) do y, x
        println("MV Product: $mv_product")
        mv_product += 1

        foreachleafnode(mesh) do level, node
            offset = nodeoffset(dofs, level, node)
            transform = nodetransform(mesh, level, node)

            block = blocks[level]

            # Transfer data to block
            transfer_to_block!(block, x, basis, mesh, dofs, level, node) do pos, face
                if faceside(face)
                    r = norm(pos)
                    return robin(1.0/r, 1.0, 0.0)
                else
                    return nuemann(1.0, 0.0)
                end
            end
            
            for (i, cell) in enumerate(cellindices(block))
                lpos = cellposition(block, cell)
                gpos = transform(lpos)

                j = inv(jacobian(transform, lpos))

                lgrad = block_gradient(block, cell, basis)
                ggrad = j * lgrad

                lhess = block_hessian(block, cell, basis)
                ghess = j' * lhess * j

                glap = ghess[1, 1] + ghess[2, 2] + ggrad[1]/gpos[1]
                gval = block_value(block, cell)

                y[offset + i] = -glap - seed[offset + i] * gval
            end
        end
    end

    println("Solving")
    solution, history = bicgstabl(hemholtz, seed, 2; log=true, max_mv_products=16000)
    @show history

    writer = MeshWriter{2, Float64}()
    attrib!(writer, NodeAttribute())
    attrib!(writer, ScalarAttribute("seed", seed))
    attrib!(writer, ScalarAttribute("solution", solution))
    write_vtu(writer, mesh, dofs, "output")
end

main()