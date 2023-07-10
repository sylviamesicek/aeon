using Aeon
using Aeon.Geometry
using Aeon.Operators
using Aeon.Methods

using IterativeSolvers
using LinearMaps
using StaticArrays

function gunlach_seed(pos::SVector{2, T}, a::T, b::Int, c::Int, σᵨ::T, σᵣ::T) where T
    r = sqrt(pos[1]^2 + pos[2]^2)

    ρ_term = (pos[1]/σᵨ)^b
    r_term = ℯ^(-(r/σᵣ)^c)

    return a * ρ_term * r_term / 4
end

# Main code
function main()
    value = LagrangeValue{Float64, 2}()
    derivative = LagrangeDerivative{Float64, 2}()
    derivative2 = LagrangeDerivative2{Float64, 2}()
    hessian = HessianFunctional{2}(value, derivative, derivative2)

    mesh = TreeMesh(HyperBox(SA[0.0, 0.0], SA{Float64}[4.0, 4.0]), 7)

    # mark_global_refine!(mesh)
    # prepare_and_execute_refinement!(mesh)

    surface = TreeSurface(mesh)

    seed = similar(surface)

    for active in surface.active
        block = TreeBlock(surface, active)
        trans = blocktransform(block)

        for cell in cellindices(block)
            lpos = cellcenter(block, cell)
            gpos = trans(lpos)

            value = gunlach_seed(gpos, 1.0, 2, 2, 1.0, 1.0)

            setfieldvalue!(cell, block, seed, value)
        end
    end

    laplacian = LinearMap(surface.total) do y, x
        yfield = TreeField{2}(y)
        xfield = TreeField{2}(x)

        for active in surface.active
            block = TreeBlock(surface, active)
            trans = blocktransform(block)

            for cell in cellindices(block)
                lpos = cellcenter(block, cell)
                # gpos = trans(lpos)
                j = inv(jacobian(trans, lpos))

                lhess = evaluate(cell, block, hessian, xfield)
                ghess = j' * lhess * j

                setfieldvalue!(cell, block, yfield, ghess[1, 1] + ghess[2, 2])
            end
        end
    end

    rhs = TreeField{2}(laplacian * seed.values)

    # # Right hand side
    # rhs = similar(surface)

    # for active in surface.active
    #     block = TreeBlock(surface, active)

    #     for cell in cellindices(block)
    #         setfieldvalue!(cell, block, rhs, 0.0)
    #     end
    # end

    hemholtz = LinearMap(surface.total) do y, x
        yfield = TreeField{2}(y)
        xfield = TreeField{2}(x)

        for active in surface.active
            block = TreeBlock(surface, active)
            trans = blocktransform(block)

            for cell in cellindices(block)
                lpos = cellcenter(block, cell)
                # gpos = trans(lpos)
                j = inv(jacobian(trans, lpos))

                lhess = evaluate(cell, block, hessian, xfield)
                ghess = j' * lhess * j

                gvalue = evaluate(cell, block, xfield)
                seed_scale = evaluate(cell, block, rhs)

                glap = ghess[1, 1] + ghess[2, 2]

                setfieldvalue!(cell, block, yfield, glap - gvalue * seed_scale)
            end
        end
    end

    Ψ = similar(surface)

    for active in surface.active
        block = TreeBlock(surface, active)
        for cell in cellindices(block)
            setfieldvalue!(cell, block, Ψ, 0.0)
        end
    end

    println("Solving")

    _, history = bicgstabl!(Ψ.values, hemholtz, rhs.values, 2; log=true, max_mv_products=4000)

    @show history

    writer = MeshWriter(surface)
    attrib!(writer, BlockAttribute())
    attrib!(writer, ScalarAttribute("seed", seed))
    attrib!(writer, ScalarAttribute("seed-laplacian", rhs))
    attrib!(writer, ScalarAttribute("Ψ", Ψ))
    write_vtu(writer, "output")
end

# # Main code
# function run_profile()
#     mesh = TreeMesh(HyperBox(SA[0.0, 0.0], SA{Float64}[π, π]), 3)

#     mark_global_refine!(mesh)
#     prepare_and_execute_refinement!(mesh)

#     surface = TreeSurface(mesh)

#     # Right hand side
#     analytic_laplacian = similar(surface)
#     analytic_value = similar(surface)

#     for active in surface.active
#         block = TreeBlock(surface, active)
#         trans = blocktransform(block)

#         for cell in cellindices(block)
#             lpos = cellcenter(block, cell)
#             gpos = trans(lpos)

#             ana = sin(gpos.x) * sin(gpos.y)
#             fun = -2sin(gpos.x)sin(gpos.y)

#             setfieldvalue!(cell, block, analytic_value, ana)
#             setfieldvalue!(cell, block, analytic_laplacian, fun)
#         end
#     end

#     # _precomp = laplacian * analytic_value.values

#     # @show length(_precomp)

#     value = LagrangeValue{Float64, 2}()
#     derivative = LagrangeDerivative{Float64, 2}()
#     derivative2 = LagrangeDerivative2{Float64, 2}()
#     hessian = HessianFunctional{2}(value, derivative, derivative2)
#     block = TreeBlock(surface, 2)

#     evaluate(CartesianIndex(4, 4), block, analytic_value)
#     evaluate(CartesianIndex(4, 4), block, (derivative, derivative), analytic_value)
#     allocated = @allocated evaluate(CartesianIndex(4, 4), block, (derivative, derivative), analytic_value)
#     println(allocated)
# end

# run_profile()

# Execute
main()