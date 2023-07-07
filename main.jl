using Aeon
using Aeon.Geometry
using Aeon.Operators
using Aeon.Methods

using StaticArrays

# Main code
function main()
    mesh = TreeMesh(HyperBox(SA[0.0, 0.0], SA{Float64}[π, π]), 6)

    # mark_global_refine!(mesh)
    # prepare_and_execute_refinement!(mesh)

    surface = TreeSurface(mesh)

    field = similar(surface)

    for active in surface.active
        block = TreeBlock(surface, active)
        trans = blocktransform(block)

        for cell in cellindices(block)
            lpos = cellcenter(block, cell)
            gpos = trans(lpos)

            fun = sin(gpos.x)* sin(gpos.y)

            setfieldvalue!(cell, block, field, fun)
        end
    end

    numeric = similar(surface)
    analytic = similar(surface)
    diff = similar(surface)

    value = LagrangeValue{Float64, 2}()
    derivative = LagrangeDerivative{Float64, 2}()

    @show cell_stencil(value)
    @show interface_value_stencil(value, 2, false)
    @show cell_stencil(derivative)

    for active in surface.active
        block = TreeBlock(surface, active)
        trans = blocktransform(block)

        for cell in cellindices(block)
            lpos = cellcenter(block, cell)
            gpos = trans(lpos)
            j = inv(jacobian(trans, lpos))

            lgrad = gradient(cell, block, value, derivative, field)
            ggrad = j * lgrad

            num = ggrad.x
            ana = cos(gpos.x) * sin(gpos.y)

            setfieldvalue!(cell, block, numeric, num)
            setfieldvalue!(cell, block, analytic, ana)
            setfieldvalue!(cell, block, diff, num - ana)
        end
    end

    writer = MeshWriter(surface)
    attrib!(writer, BlockAttribute())
    attrib!(writer, ScalarAttribute("function", field))
    attrib!(writer, ScalarAttribute("numeric", numeric))
    attrib!(writer, ScalarAttribute("analytic", analytic))
    attrib!(writer, ScalarAttribute("error", diff))
    write_vtu(writer, "output")
end

# Execute
main()