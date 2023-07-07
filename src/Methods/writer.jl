## Exports
export BlockAttribute, ScalarAttribute, MeshWriter, attrib!, write_vtu

using WriteVTK

"""
Adds `builtin::blockid` data to VTK file.
"""
struct BlockAttribute end

struct ScalarAttribute{N, T}
    name::String
    field::TreeField{N, T}

    ScalarAttribute(name::String, field::TreeField{N, T}) where {N, T} = new{N, T}(name, field)
end

mutable struct MeshWriter{N, T}
    surface::TreeSurface{N, T}
    blocks::Bool
    fields::Vector{ScalarAttribute{N, T}}
end

MeshWriter(surface::TreeSurface{N, T}) where {N, T} = MeshWriter{N, T}(surface, false, [])

function attrib!(writer::MeshWriter, attrib::ScalarAttribute)
    push!(writer.fields, attrib)
end

function attrib!(writer::MeshWriter, ::BlockAttribute)
    writer.blocks = true
end

function write_vtu(writer::MeshWriter{N, T}, filename::String) where {N, T}
    surface = writer.surface

    positions = Vector{SVector{N, T}}()
    cells = Vector{MeshCell}()

    point_offset = 0

    for active in surface.active
        block = TreeBlock(surface, active)
        trans = blocktransform(block)
        dims = blockcells(block)

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

    position_matrix = Matrix{T}(undef, N, length(positions))

    for i in eachindex(positions)
        for j in 1:N
            position_matrix[j, i] = positions[i][j]
        end
    end

    # Produce VTK Grid
    vtk_grid(filename, position_matrix, cells) do vtk
        if writer.blocks
            blocks = Vector{T}(undef, surface.total)

            for active in surface.active
                block = TreeBlock(surface, active)
                offset = surface.offsets[active]

                for (i, _) in enumerate(cellindices(block))
                    blocks[offset + i] = active
                end
            end

            vtk["builtin:blocks", VTKCellData()] = blocks
        end
        
        # Scalars
        for i in eachindex(writer.fields)
            name = writer.fields[i].name
            field = writer.fields[i].field

            @assert length(field.values) == surface.total

            vtk["scalar:$(name)", VTKCellData()] = field.values
        end
    end
end