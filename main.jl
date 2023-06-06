using Aeon
using Aeon.Analytic
using Aeon.Method
using Aeon.Grid

using LinearAlgebra
using StaticArrays

# Main code
function main()
    # basis = tensor_basis(1, 4)
    # weight = Gaussian(1.0, 1.0)
    # engine = WLSEngine{1, Float64}(basis, weight)

    # x = unknown()
    # operator =  SVector(1.0) ⋅ ∇(x)
    # # operator =  1.0x
    # vertex = uniform_grid_vertex(1, Float64, 3)

    # sten = stencil(engine, vertex, operator)

    # display(sten)

    grid = GridMesh(SVector(0.0, 0.0), SVector(10, 10), 0.1, 2)
    level = last(grid)

    field = similar(level)

    for i in eachindex(level)
        field[i] = level[i][1]
    end
    

    writer = GridWriter(level)
    attrib!(writer, IndexAttribute())
    attrib!(writer, ScalarAttribute("test", field))
    write_vtk(writer, "output")
end

main()