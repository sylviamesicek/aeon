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

function solve_nth_order(mesh::Mesh{N, T}, dofs::DoFManager{N, T}, basis::AbstractBasis{T}, amp::T, ::Val{O}) where {N, T, O}
    seed = project(mesh, dofs) do pos
        initial_data(pos, amp, one(T), zero(T), zero(T), zero(T))
    end

    blocks = BlockManager{O}(mesh)
    total = dofstotal(dofs)

    # mv_product = 1

    println("Solving to order $(O)")

    hemholtz = LinearMap(total) do y, x
        # println("MV Product: $mv_product")
        # mv_product += 1

        foreachleafnode(mesh) do level, node
            offset = nodeoffset(dofs, level, node)
            transform = nodetransform(mesh, level, node)

            block = blocks[level]

            # Transfer data to block
            transfer_to_block!(block, x, basis, mesh, dofs, level, node) do pos, face
                if faceside(face)
                    r = norm(pos)
                    return robin(one(T)/r, one(T), zero(T))
                else
                    return nuemann(one(T), zero(T))
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

    solution, history = bicgstabl(hemholtz, seed, 2; log=true, max_mv_products=16000 * O)
    @show history

    return solution
end

using DelimitedFiles

# Main code
function main()
    # Function basis
    basis = LagrangeBasis{Float64}()

    # Mesh

    ratios = Vector{Tuple{Float64, Float64}}()

    Threads.@threads for A in 1.0:.1:2.0
        coarsemesh = Mesh(HyperBox(SA[0.0, 0.0], SA[4.0, 4.0]), 4, 0)
        refinedmesh = Mesh(HyperBox(SA[0.0, 0.0], SA[4.0, 4.0]), 5, 0)

        coarsedofs = DoFManager(coarsemesh)
        refinedofs = DoFManager(refinedmesh)

        println("A = $A")
        coarsesol4 = solve_nth_order(coarsemesh, coarsedofs, basis, A, Val(2))
        coarsesol8 = solve_nth_order(coarsemesh, coarsedofs, basis, A, Val(4))

        refinedsol4 = solve_nth_order(refinedmesh, refinedofs, basis, A, Val(2))
        refinedsol8 = solve_nth_order(refinedmesh, refinedofs, basis, A, Val(4))

        coarseerror = maximum(coarsesol4 .- coarsesol8)
        refinederror = maximum(refinedsol4 .- refinedsol8)

        ratio = refinederror / coarseerror

        push!(ratios, (ratio, A))
    end

    writedlm("ratios.csv",  ratios, ',')

    println("Ratio: {ratios}")

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

    # dofs = DoFManager(mesh)
    # total = dofstotal(dofs)

    # @show mesh
    # @show total

    # seed = project(mesh, dofs) do pos
    #     initial_data(pos, 1.0, 1.0, 0.0, 0.0, 0.0)
    # end

    # sol4 = solve_nth_order(mesh, dofs, basis, Val(2))
    # sol8 = solve_nth_order(mesh, dofs, basis, Val(4))

    # writer = MeshWriter{2, Float64}()
    # attrib!(writer, NodeAttribute())
    # attrib!(writer, ScalarAttribute("seed", seed))
    # attrib!(writer, ScalarAttribute("solution-4th", sol4))
    # attrib!(writer, ScalarAttribute("solution-8th", sol8))
    # attrib!(writer, ScalarAttribute("solution-diff", sol4 .- sol8))
    # write_vtu(writer, mesh, dofs, "output")
end

main()