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

    mesh = Mesh(HyperBox(SA[0.0, 0.0], SA[4.0, 4.0]), 7)

    mark_refine_global!(mesh)
    prepare_and_execute_refinement!(mesh)
    
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

    seed = project(mesh, dofs) do pos
        gunlach_laplacian(pos, 1.0, 1.0)
    end

    block = ArrayBlock{Float64, 2}(undef, nodecells(mesh)...)

    mv_product = 1

    hemholtz = LinearMap(total) do y, x
        println("MV Product: $mv_product")
        mv_product += 1

        for level in eachindex(mesh)
            for node in eachleafnode(mesh, level)
                offset = nodeoffset(dofs, level, node)
                transform = nodetransform(mesh, level, node)

                # Transfer data to block
                transfer_to_block!(block, basis, mesh, dofs, level, node, x) do boundary, axis
                    if boundary_edge(boundary, axis) > 0
                        # Robin Conditions
                        lpos = cellposition(block, boundary.cell)
                        gpos = transform(lpos)
                        r = norm(gpos)

                        robin(1.0/r, 1.0, 0.0)
                    else
                        # Nuemann Conditions
                        nuemann(1.0, 0.0)
                    end
                end
                
                for (i, cell) in enumerate(cellindices(block))
                    lpos = cellposition(block, cell)
                    gpos = transform(lpos)

                    j = inv(jacobian(transform, lpos))

                    lgrad = blockgradient(block, cell, basis)
                    ggrad = j * lgrad

                    lhess = blockhessian(block, cell, basis)
                    ghess = j' * lhess * j

                    glap = ghess[1, 1] + ghess[2, 2] + ggrad[1]/gpos[1]
                    gval = blockvalue(block, cell)

                    y[offset + i] = -glap - seed[offset + i] * gval
                end
            end
        end
    end

    println("Solving")

    # solution, history = bicgstabl(hemholtz, seed, 2; log=true, max_mv_products=8000)

    # @show history

    solution = hemholtz * seed

    writer = MeshWriter{2, Float64}()
    attrib!(writer, NodeAttribute())
    attrib!(writer, ScalarAttribute("seed", seed))
    attrib!(writer, ScalarAttribute("solution", solution))
    write_vtu(writer, mesh, dofs, "output")
end

struct PoissonOperator{N, T, O} <: AbstractOperator{T}
    mesh::Mesh{N, T}
    dofs::DoFManager{N, T}
    blocks::BlockManager{N, T, O}
end

function operator_apply!(y::AbstractVector{T}, oper::PoissonOperator{N, T}, x::AbstractVector{T}, maxlevel::Int) where {N, T}
    basis = LagrangeBasis{T}()

    foreachleafnode(mesh, maxlevel) do level, node
        offset = nodeoffset(oper.dofs, level, node)
        transform = nodetransform(oper.mesh, level, node)
        block = oper.blocks[level]

        # Transfer data to block
        transfer_to_block!(block, basis, oper.mesh, oper.dofs, level, node, x) do boundary, axis
            diritchlet(1.0, 0.0)
        end

        for (i, cell) in enumerate(cellindices(block))
            lpos = cellposition(block, cell)
            j = inv(jacobian(transform, lpos))

            lhess = blockhessian(oper.block, cell, basis)
            ghess = j' * lhess * j

            glap = ghess[1, 1] + ghess[2, 2]

            y[offset + i] = -glap
        end
    end
end

function operator_diagonal!(y::AbstractVector{T}, oper::PoissonOperator{N, T}, maxlevel::Int) where {N, T}
    basis = LagrangeBasis{T}()

    foreachleafnode(mesh, maxlevel) do level, node
        offset = nodeoffset(oper.dofs, level, node)
        transform = nodetransform(oper.mesh, level, node)
        block = oper.blocks[level]

        for (i, cell) in enumerate(cellindices(block))
            lpos = cellposition(block, cell)
            j = inv(jacobian(transform, lpos))

            lhess = hessian_diagonal(Val(N), basis)
            ghess = j' * lhess * j

            glap = ghess[1, 1] + ghess[2, 2]

            y[offset + i] = -glap
        end
    end
end


function operator_restrict!(y::AbstractVector{T}, oper::PoissonOperator{N, T}, x::AbstractVector{T}, maxlevel::Int) where {N, T}
    basis = LagrangeBasis{T}()

    if maxlevel == 1
        return
    end

    for level in 1:(maxlevel - 1)
        for node in eachnode(oper.mesh, level)
            if nodechildren(oper.mesh, level) == -1
                offset = nodeoffset(oper.dofs, level, node)
                block = oper.blocks[level]

                for (i, cell) in enumerate(cellindices(block))
                    y[offset + i] = x[offset + i]
                end

            end
        end
    end

    if maxlevel ≤ baselevels(oper.mesh)
        # Restrict to single parent
        for node in eachnode(oper.mesh, maxlevel)
            parent = nodeparent(oper.mesh, maxlevel, node)
            poffset = nodeoffset(oper.dofs, maxlevel-1, parent)
            block = oper.blocks[maxlevel]

            transfer_to_block!(block, basis, oper.mesh, oper.dofs, level, node, x) do boundary, axis
                diritchlet(1.0, 0.0)
            end

            for (i, pcell) in enumerate(CartesianIndices(blockcells(block) ./ 2))
                vertex = map(VertexIndex, pcell.I .* 2)
                y[poffset + i] = block_prolong(block, vertex, basis)
            end
        end
        
    else
        # Restrict to subcell of parent
        error("Restriction to node parents is unimplemented")
    end
end

# function operator_prolong!(y::AbstractVector{T}, oper::AbstractOperator{T}, x::AbstractVector{T}, maxlevel::Int) where {N, T}

# end

# Main code
function main2()
    # Function basis
    basis = LagrangeBasis{Float64}()

    # Mesh

    mesh = Mesh(HyperBox(SA[0.0, 0.0], SA{Float64}[π, π]), 7, 0)

    # mark_refine_global!(mesh)
    # prepare_and_execute_refinement!(mesh)
    # mark_refine_global!(mesh)
    # prepare_and_execute_refinement!(mesh)

    dofs = DoFManager(mesh)
    total = dofstotal(dofs)

    @show mesh
    @show total

    seed = project(mesh, dofs) do pos
        sin(pos.x)*sin(pos.y)
        # 1.0
    end

    analytic = project(mesh, dofs) do pos
        2sin(pos.x)*sin(pos.y)
        # 0.0
    end

    # analytic = project(mesh, dofs) do pos
    #     # sin(pos.x)*sin(pos.y)/2
    #     1.0
    # end

    blocks = BlockManager{2}(mesh)

    mv_product = 1

    hemholtz = LinearMap(total) do y, x
        println("MV Product: $mv_product")
        mv_product += 1

        foreachleafnode(mesh) do level, node
            block = blocks[level]
            offset = nodeoffset(dofs, level, node)
            transform = nodetransform(mesh, level, node)

            fill!(block, 0.0)
            # Transfer data to block
            transfer_to_block!(block, x, mesh, dofs, basis, level, node) do boundary, axis
                diritchlet(1.0, 0.0)
            end
            
            for (i, cell) in enumerate(cellindices(block))
                lpos = cellposition(block, cell)
                j = inv(jacobian(transform, lpos))
                lhess = block_hessian(block, cell, basis)
                ghess = j' * lhess * j
                glap = ghess[1, 1] + ghess[2, 2]
                y[offset + i] = -glap
            end
        end
    end

    println("Solving")

    solution, history = bicgstabl(hemholtz, seed, 2; log=true, max_mv_products=1000)

    @show history

    application = hemholtz * seed

    writer = MeshWriter{2, Float64}()
    attrib!(writer, NodeAttribute())
    attrib!(writer, ScalarAttribute("seed", seed))
    attrib!(writer, ScalarAttribute("solution", solution))
    attrib!(writer, ScalarAttribute("application", application))
    attrib!(writer, ScalarAttribute("error", application .- analytic))
    write_vtu(writer, mesh, dofs, "output")
end

# Execute
main2()