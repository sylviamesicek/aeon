using Aeon
using Aeon.Geometry
using Aeon.Operators
using Aeon.Methods

using StaticArrays

# Main code
function main()
    mesh = TreeMesh(HyperBox(SA[0.0, 0.0], SA{Float64}[π, π]), 4)

    mark_global_refine!(mesh)
    prepare_and_execute_refinement!(mesh)

    surface = TreeSurface(mesh)

    field1 = similar(surface)

    for active in surface.active
        block = TreeBlock(surface, active)
        trans = blocktransform(block)

        for cell in cellindices(block)
            lpos = cellcenter(block, cell)
            gpos = trans(lpos)

            setfieldvalue!(cell, block, field1, sin(gpos.x)*sin(gpos.y) )
        end
    end

    field2 = similar(surface)

    opers = (LagrangeDerivative{Float64, 2}(), LagrangeValue{Float64, 2}())

    for active in surface.active
        block = TreeBlock(surface, active)
        trans = blocktransform(block)

        for cell in cellindices(block)
            lpos = cellcenter(block, cell)
            gpos = trans(lpos)

            value = evaluate(cell, block, opers, field1)

            setfieldvalue!(cell, block, field2, value)
        end
    end

    writer = MeshWriter(surface)
    attrib!(writer, BlockAttribute())
    attrib!(writer, ScalarAttribute("field1", field1))
    attrib!(writer, ScalarAttribute("field2", field2))
    write_vtu(writer, "output")
end

# Execute
main()