using Aeon
using Aeon.Space
using StaticArrays

# Main code
function main()
    grid = Grid(SVector(-1.0, -1.0), SVector(1.0, 1.0), SVector{2, UInt}(100, 100))
    domain = mkdomain(grid)

    # count = length(domain)

    # func = zeros(count)

    # for i in eachindex(domain)
    #     func[i] = domain[i][1]
    # end

    # writer = DomainWriter{2, Float64}(domain)
    # attrib!(writer, ScalarAttribute("test", func))
    # write_vtk(writer, "output")
end

main()