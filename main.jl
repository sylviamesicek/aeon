using Aeon
using Aeon.Analytic
using Aeon.Method
using Aeon.Grid

using LinearAlgebra
using StaticArrays

# Main code
function main()
    method = GridMethod(SVector(0.0, 0.0), 1.0, SVector(4, 4), 2)
    mesh = finest(method)

    @show mesh

    field = similar(mesh)

    for i in eachindex(mesh)
        field[i] = mesh[i][1]
    end
    
    writer = MeshWriter(mesh)
    attrib!(writer, IndexAttribute())
    attrib!(writer, KindAttribute())
    attrib!(writer, ScalarAttribute("test", field))
    write_vtk(writer, "output")
end

main()