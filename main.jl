using Aeon
using Aeon.Space
using StaticArrays

# Main code
function main()
    grid = Grid{2, Float64}(SVector{2}([-1.0, -1.0]), SVector{2}([1.0, 1.0]), SVector{2, UInt}([100, 100]))
    domain = mkdomain(grid)

    count = length(domain)

    func = zeros(count)

    for i in eachindex(domain)
        position = domain[i].position
        func[i] = sin(position[1]) + cos(position[2])
    end

    writer = DomainWriter{2, Float64}(domain)
    attrib!(writer, ScalarAttribute("test", func))
    write_vtk(writer, "output")
end

main()