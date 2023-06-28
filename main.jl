using Aeon
using Aeon.Geometry
using Aeon.Methods
using Aeon.Operators

using LinearAlgebra
using StaticArrays

# Main code
function main()
    # mesh = hyperprism(SA[0.0, 0.0], SA[1.0, 1.0], SA[4, 4], 3)
    
    # @show mesh.doftotal

    # for _ in 1:3
    #     refine!(mesh) do cell
    #         cent = center(cell.bounds)
    #         norm(cent) ≤ 1.0
    #     end
    # end

    # @show mesh.doftotal


    # writer = MeshWriter(mesh)
    # attrib!(writer, IndexAttribute())
    # attrib!(writer, CellAttribute())
    # write_vtk(writer, "output")

    source = MattssonNordström2004{Float64, 1}()

    derivative = derivative_operator(source, Val(1))
    prolong = prolongation_operator(source)

    field = Matrix{Float64}(undef, 101, 101)

    point = CartesianIndex(50, 50)

    product(point, (derivative, prolong), field)
end

# Execute
main()