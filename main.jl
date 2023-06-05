using Aeon
using Aeon.Analytic
using Aeon.Method

using LinearAlgebra
using StaticArrays

# Main code
function main()
    basis = tensor_basis(1, 4)
    weight = Gaussian(1.0, 1.0)
    engine = WLSEngine{1, Float64}(basis, weight)

    x = unknown()
    operator =  SVector(1.0) ⋅ ∇(x)
    # operator =  1.0x
    vertex = uniform_grid_vertex(1, Float64, 3)

    sten = stencil(engine, vertex, operator)

    display(sten)
    

    

    # domain = Domain(Grid(SVector(-1.0, -1.0), SVector(1.0, 1.0), SVector{2, UInt}(3, 3)))

    # basis = tensor_basis(2, 1)
    # weight = Gaussian(1.0, 1.0)

    # mesh = WLSMesh{4}(domain, basis, weight)

    # system = meshmatrix(mesh)

    # display(system)

    # count = length(domain)

    # func = zeros(count)

    # for i in eachindex(domain)
    #     func[i] = domain[i][1]
    # end

    # writer = DomainWriter(domain)
    # # attrib!(writer, ScalarAttribute("test", func))
    # attrib!(writer, NormalAttribute())
    # attrib!(writer, IndexAttribute())
    # write_vtk(writer, "output")

    # x = unknown()

    # display(2x + SVector(1.0, 1.0) ⋅ ∇(x) + Δ(x))
end

main()