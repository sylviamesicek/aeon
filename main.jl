using Aeon
using Aeon.Geometry
using Aeon.Operators
using Aeon.Methods

using IterativeSolvers
using LinearMaps
using StaticArrays

# Main code
function main()
    mesh = TreeMesh(HyperBox(SA[0.0, 0.0], SA{Float64}[π, π]), 4)

    mark_global_refine!(mesh)
    prepare_and_execute_refinement!(mesh)

    surface = TreeSurface(mesh)

    # Right hand side
    rhs = similar(surface)

    for active in surface.active
        block = TreeBlock(surface, active)
        trans = blocktransform(block)

        for cell in cellindices(block)
            lpos = cellcenter(block, cell)
            gpos = trans(lpos)

            fun = -2sin(gpos.x)sin(gpos.y)

            setfieldvalue!(cell, block, rhs, fun)
        end
    end

    # Set analytic solution
    analytic = similar(surface)

    for active in surface.active
        block = TreeBlock(surface, active)
        trans = blocktransform(block)

        for cell in cellindices(block)
            lpos = cellcenter(block, cell)
            gpos = trans(lpos)

            ana = sin(gpos.x) * sin(gpos.y)
            setfieldvalue!(cell, block, analytic, ana)
        end
    end

    value = LagrangeValue{Float64, 2}()
    derivative = LagrangeDerivative{Float64, 2}()
    derivative2 = LagrangeDerivative2{Float64, 2}()

    hessian = HessianFunctional{2}(value, derivative, derivative2)

    operator = LinearMap(surface.total) do y, x
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

    solvalues, history = bicgstabl(operator, rhs.values, 2; log=true)
    # solvalues = operator * analytic.values
    solution = TreeField{2}(solvalues)

    @show history

    error = TreeField{2}(solution.values .- analytic.values)

    writer = MeshWriter(surface)
    attrib!(writer, BlockAttribute())
    attrib!(writer, ScalarAttribute("right-hand-side", rhs))
    attrib!(writer, ScalarAttribute("sol-numeric", solution))
    attrib!(writer, ScalarAttribute("sol-analytic", analytic))
    attrib!(writer, ScalarAttribute("error", error))
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