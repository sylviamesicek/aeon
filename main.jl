using Aeon
using Aeon.Geometry
using Aeon.Methods
using Aeon.Operators

using LinearAlgebra
using StaticArrays

struct Poisson{N, T} end

function apply_operator!(result::AbstractVector{T}, x::AbstractVector{T}, mesh::Mesh{N, T}, ::Poisson{N, T}) where {N, T}
    fill!(result, 0)

    source = MattssonNordström2004{T, 2}()

    d1 = derivative_operator(source, Val(1))
    d2 = derivative_operator(source, Val(2))

    prolong = prolongation_operator(source)
    restrict = restriction_operator(source)

    hessian = Hessian{N}(d1, d2)

    interface_strength = 0
    boundary_strength = 0

    # Main equation
    for cell in CartesianIndices(size(mesh.cells))
        # Subfields
        cellx = cellfield(mesh[cell], x)
        cellresult = cellfield(mesh[cell], result)
        # Operator
        for point in eachindex(mesh[cell])
            j = Aeon.Methods.jacobian(mesh[cell], point)
            hess = transform_hessian(j, evaluate(point, hessian, cellx))
            # Main Equation
            cellresult[point] += hess[1, 1] + hess[2, 2]

            # Boundary/Interface Conditions
            for axis in 1:N
                if !(point[axis] == 1 || point[axis] == size(cellx)[axis])
                    continue
                end

                direction = ifelse(point[axis] == 1, -1, 1)

                # oper_identity = ntuple(dim -> IdentityOperator{T, 2}(), Val(N))
                oper_prolong = ntuple(dim -> dim == axis ? IdentityOperator{T, 2}() : prolong, Val(N))
                oper_restrict = ntuple(dim -> dim == axis ? IdentityOperator{T, 2}() : restrict, Val(N))

                other = CartesianIndex(ntuple(Val(N)) do dim
                    ifelse(dim == axis, cell[axis] + direction, cell[dim])
                end)

                if other[axis] > 0 && other[axis] < size(mesh.cells)[axis]
                    # Interface Conditions
                    otherx = cellfield(mesh[other], x)
                    otheredge = point[axis] == 1 ? size(otherx)[axis] : 1
                    otherpoint = CartesianIndex(ntuple(dim -> ifelse(dim == axis, otheredge, point[dim]), Val(N)))

                    value = cellx[point]

                    if mesh[cell].refinement == mesh[other].refinement
                        othervalue = otherx[otherpoint]
                    elseif mesh[cell].refinement < mesh[other].refinement
                        othervalue = product(otherpoint, oper_restrict, otherx)
                    else
                        othervalue = product(otherpoint, oper_prolong, otherx)
                    end

                    cellresult[point] += interface_strength * (value - othervalue)
                else
                    value = cellx[point]
                    # Boundary Condition
                    cellresult[point] += boundary_strength * value
                end
            end
        end
    end
end

function compute_rhs(rhs::AbstractVector{T}, ::Mesh{N, T}, ::Poisson{N, T}) where {N, T}
    fill!(rhs, 0)
end

# Main code
function main()
    mesh = hyperprism(SA[0.0, 0.0], SA[1.0, 1.0], (4, 4), 4)
    
    @show mesh.doftotal

    for _ in 1:3
        refine!(mesh) do cell
            cent = center(mesh[cell].bounds)
            norm(cent) ≤ 1.0
        end
    end

    @show mesh.doftotal

    field = Vector{Float64}(undef, mesh.doftotal)
    result = Vector{Float64}(undef, mesh.doftotal)

    for cell in CartesianIndices(size(mesh.cells))
        cellf = cellfield(mesh[cell], field)

        for point in eachindex(mesh[cell])
            pos = Aeon.Methods.position(mesh[cell], point)
            cellf[point] = sum(pos .^ 2)
        end
    end

    # source = MattssonNordström2004{Float64, 2}()
    # d2 = derivative_operator(source, Val(2))
    # laplace = Laplacian{2}(d2)

    # for cell in CartesianIndices(size(mesh.cells))
    #     # Subfields
    #     cellf = cellfield(mesh[cell], field)
    #     cellresult = cellfield(mesh[cell], result)
    #     # Operator
    #     for point in eachindex(mesh[cell])
    #         cellresult[point] = evaluate(point, laplace, cellf)
    #     end
    # end

    apply_operator!(result, field, mesh, Poisson{2, Float64}())

    # source = MattssonNordström2004{Float64, 2}()

    # d1 = derivative_operator(source, Val(1))
    # d2 = derivative_operator(source, Val(2))

    # laplace = Laplacian{1}(d2)

    # @show value = evaluate(CartesianIndex(20), laplace, field)

    writer = MeshWriter(mesh)
    attrib!(writer, IndexAttribute())
    attrib!(writer, CellAttribute())
    attrib!(writer, ScalarAttribute{2}("field", field))
    attrib!(writer, ScalarAttribute{2}("result", result))
    write_vtk(writer, "output")
end

# Execute
main()