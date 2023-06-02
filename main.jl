using Aeon
using Aeon.Analytic
using Aeon.Space
using Aeon.Engine

using StaticArrays

# Main code
function main()
    domain = Domain(Grid(SVector(-1.0, -1.0), SVector(1.0, 1.0), SVector{2, UInt}(3, 3)))

    basis = tensor_basis(2, 1)
    weight = Gaussian(1.0, 1.0)

    mesh = WLSMesh{4}(domain, basis, weight)

    system = meshmatrix(mesh)

    display(system)

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
end

main()