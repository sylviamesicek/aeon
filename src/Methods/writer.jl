## Exports
export NodeAttribute, ScalarAttribute, MeshWriter, attrib!, write_vtu

using WriteVTK

"""
Adds `builtin::node` data to VTK file.
"""
struct NodeAttribute end

struct ScalarAttribute{T}
    name::String
    field::Vector{T}

    ScalarAttribute(name::String, field::Vector{T}) where {T} = new{T}(name, field)
end

mutable struct MeshWriter{N, T}
    blocks::Bool
    fields::Vector{ScalarAttribute{T}}
end

MeshWriter{N, T}() where {N, T} = MeshWriter{N, T}(false, [])

function attrib!(writer::MeshWriter, attrib::ScalarAttribute)
    push!(writer.fields, attrib)
end

function attrib!(writer::MeshWriter, ::NodeAttribute)
    writer.blocks = true
end

function write_vtu(writer::MeshWriter{N, T}, mesh::Mesh{N, T}, dofs::DoFHandler{N, T}, filename::String) where {N, T}
    dims = nodecells(mesh)
    total = dofstotal(dofs)
    
    positions = SVector{N, T}[]
    cells = MeshCell[]

    point_offset = 0

    for level in eachindex(mesh)
        for leaf in eachleafnode(mesh, level)
            trans = nodetransform(mesh, level, leaf)

            tolinear = LinearIndices(dims .+ 1)
            points_full = CartesianIndices(dims .+ 1)
            points_cell = CartesianIndices(dims)
    
            point_total = prod(dims .+ 1)
    
            for point in points_full
                locposition = SVector{N, T}((point.I .- 1) ./ dims)
                gloposition = trans(locposition)
    
                push!(positions, gloposition)
            end
    
            if N == 1
                for point in points_cell
                    p = point.I
                    v1 = tolinear[p...]
                    v2 = tolinear[(p .+ 1)...]
                    push!(cells, MeshCell(VTKCellTypes.VTK_LINE, (v1, v2) .+ point_offset))
                end 
            elseif N == 2
                for point in points_cell
                    p = point.I
                    v1 = tolinear[p...]
                    v2 = tolinear[(p .+ (0, 1))...]
                    v3 = tolinear[(p .+ (1, 1))...]
                    v4 = tolinear[(p .+ (1, 0))...]
                    push!(cells, MeshCell(VTKCellTypes.VTK_QUAD, (v1, v2, v3, v4) .+ point_offset))
                end 
            elseif N == 3
                for point in points_cell
                    p = point.I
                    v1 = tolinear[p...]
                    v2 = tolinear[(p .+ (0, 1, 0))...]
                    v3 = tolinear[(p .+ (1, 1, 0))...]
                    v4 = tolinear[(p .+ (1, 0, 0))...]
                    v5 = tolinear[(p .+ (1, 0, 1))...]
                    v6 = tolinear[(p .+ (0, 0, 1))...]
                    v7 = tolinear[(p .+ (0, 1, 1))...]
                    v8 = tolinear[(p .+ (1, 1, 1))...]
                    push!(cells, MeshCell(VTKCellTypes.VTK_HEXAHEDRON, (v1, v2, v3, v4, v5, v6, v7, v8) .+ point_offset))
                end
            else
                error("N must be ≤ 3.")
            end    
            
            point_offset += point_total
        end
    end

    position_matrix = Matrix{T}(undef, N, length(positions))

    for i in eachindex(positions)
        for j in 1:N
            position_matrix[j, i] = positions[i][j]
        end
    end

    # Produce VTK Grid
    vtk_grid(filename, position_matrix, cells) do vtk
        if writer.blocks
            blocks = Vector{T}(undef, total)

            for level in eachindex(mesh)
                for leaf in eachleafnode(mesh, level)
                    offset = nodeoffset(dofs, level, leaf)
                    for (i, _) in enumerate(CartesianIndices(dims))
                        blocks[offset + i] = leaf
                    end
                end
            end

            vtk["builtin:nodes", VTKCellData()] = blocks
        end
        
        # Scalars
        for i in eachindex(writer.fields)
            name = writer.fields[i].name
            field = writer.fields[i].field

            @assert length(field) == total

            vtk["scalar:$(name)", VTKCellData()] = field
        end
    end
end