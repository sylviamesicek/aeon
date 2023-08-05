const std = @import("std");

pub const VtkCellType = enum {
    quad,
    hexa,

    pub fn dimension(self: VtkCellType) usize {
        return switch (self) {
            .quad => 2,
            .hexa => 3,
        };
    }

    pub fn vertices(self: VtkCellType) usize {
        return switch (self) {
            .quad => 4,
            .hexa => 8,
        };
    }

    pub fn tag(self: VtkCellType) i8 {
        return switch (self) {
            .quad => 9,
            .hexa => 12,
        };
    }
};

/// A vtk data array backed by an array of doubles.
pub const VtkDataArray = struct {
    name: []const u8,
    data: std.ArrayListUnmanaged(f64),
    n_components: usize,
};

const VtkDataArrayList = std.ArrayListUnmanaged(VtkDataArray);

pub const VtkGrid = struct {
    allocator: std.mem.Allocator,
    point_data: VtkDataArrayList,
    cell_data: VtkDataArrayList,

    pub fn init(allocator: std.mem.Allocator) VtkGrid {
        return .{
            .allocator = allocator,
            .point_data = VtkDataArrayList{},
            .cell_data = VtkDataArrayList{},
        };
    }

    pub fn deinit(self: *VtkGrid) void {
        self.point_data.deinit(self.allocator);
        self.cell_data.deinit(self.allocator);
    }

    pub fn add_field(self: *VtkGrid, name: []const u8, data: []const f64, dimension: i32) !void {
        return self.point_data.append(self.allocator, .{
            .name = name,
            .data = data,
            .n_components = dimension,
        });
    }

    pub fn add_cell_field(self: *VtkGrid, name: []const u8, data: []const f64, dimension: i32) !void {
        return self.cell_data.append(self.allocator, .{
            .name = name,
            .data = data,
            .n_components = dimension,
        });
    }

    pub fn add_scalar_field(self: *VtkGrid, name: []const u8, data: []const f64) !void {
        return self.add_field(name, data, 1);
    }

    pub fn add_cell_scalar_field(self: *VtkGrid, name: []const u8, data: []const f64) !void {
        return self.add_cell_field(name, data, 1);
    }

    pub fn add_vector_field(self: *VtkGrid, name: []const u8, data: []const f64, dimension: i32) !void {
        return self.add_field(name, data, dimension);
    }

    pub fn add_cell_vector_field(self: *VtkGrid, name: []const u8, data: []const f64, dimension: i32) !void {
        return self.add_cell_field(name, data, dimension);
    }

    pub fn write_unstructured(self: VtkGrid, cell_type: VtkCellType, points: []const f64, connectivity: []const i64, out_stream: anytype) @TypeOf(out_stream).Error!void {
        const dimension = cell_type.dimension();
        const n_points: usize = points.len / dimension;

        const vertices = cell_type.vertices();
        const n_cells: usize = connectivity.len / vertices;

        try write_header(n_points, n_cells, out_stream);
        try write_points(dimension, points, out_stream);
        try write_cells(cell_type, connectivity, out_stream);
        try self.write_point_data(out_stream);
        try self.write_cell_data(out_stream);
        try write_footer(out_stream);
    }

    fn write_point_data(self: VtkGrid, out_stream: anytype) @TypeOf(out_stream).Error!void {
        try out_stream.print("<PointData>\n", .{});
        for (self.point_data.items) |point_data| {
            try write_data_array(point_data, out_stream);
        }
        try out_stream.print("</PointData>\n", .{});
    }

    fn write_cell_data(self: VtkGrid, out_stream: anytype) @TypeOf(out_stream).Error!void {
        try out_stream.print("<CellData>\n", .{});
        for (self.point_data.items) |point_data| {
            try write_data_array(point_data, out_stream);
        }
        try out_stream.print("</CellData>\n", .{});
    }

    fn write_header(n_points: usize, n_cells: usize, out_stream: anytype) @TypeOf(out_stream).Error!void {
        return out_stream.print(
            \\<VTKFile type="UnstructuedGrid" version="1.0">
            \\<UnstructuredGrid>
            \\<Piece NumberOfPoints="{}" NumberOfCells="{}">
            \\
        , .{ n_cells, n_points });
    }

    fn write_footer(out_stream: anytype) @TypeOf(out_stream).Error!void {
        return out_stream.print(
            \\</Piece>
            \\</UnstructuredGrid>
            \\</VTKFile>
            \\
        , .{});
    }

    fn write_points(n_components: usize, points: []const f64, out_stream: anytype) @TypeOf(out_stream).Error!void {
        try out_stream.print(
            \\<Points>
            \\<DataArray type="Float64" NumberOfComponents="{}" format="ascii">
            \\
        , .{n_components});

        try write_vec_array(f64, n_components, points, out_stream);

        try out_stream.print(
            \\</DataArray>
            \\</Points>
            \\
        , .{});
    }

    fn write_cells(cell_type: VtkCellType, cells: []const i64, out_stream: anytype) @TypeOf(out_stream).Error!void {
        const n_vertices = cell_type.vertices();
        const tag = cell_type.tag();

        // Connectivity Info
        try out_stream.print(
            \\<Cells>
            \\<DataArray type="Int64" Name="connectivity" NumberOfComponents="{}" format="ascii">
            \\
        , .{n_vertices});

        try write_vec_array(i64, n_vertices, cells, out_stream);

        try out_stream.print("</DataArray>\n", .{});

        // Tags
        try out_stream.print(
            \\<DataArray type="Int64" Name="types" format="ascii" RangeMin="{}" RangeMax="{}">
            \\
        , .{ tag, tag });

        const n_cells: usize = cells.len / n_vertices;

        for (0..n_cells) |_| {
            try out_stream.print("{}\n", .{tag});
        }

        try out_stream.print("</DataArray>\n", .{});

        // Offsets
        try out_stream.print(
            \\<DataArray type="Int64" Name="offsets" format="ascii" RangeMin="{}" RangeMax="{}">
            \\
        , .{ n_vertices, n_cells * n_vertices });

        var acc: usize = n_vertices;
        for (0..n_cells) |_| {
            try out_stream.print("{}\n", .{acc});
            acc += n_vertices;
        }

        try out_stream.print(
            \\</DataArray>
            \\</Cells>
            \\
        , .{});
    }

    fn write_data_array(array: VtkDataArray, out_stream: anytype) @TypeOf(out_stream).Error!void {
        try out_stream.print(
            \\<DataArray type="Float64" Name="{s}" NumberOfComponents="{}" format="ascii">
            \\
        , .{ array.name, array.n_components });
        try write_vec_array(f64, array.n_components, array.data.items, out_stream);
        try out_stream.print("</DataArray>\n", .{});
    }

    fn write_vec_array(comptime T: type, n_components: usize, data: []const T, out_stream: anytype) @TypeOf(out_stream).Error!void {
        const n_vecs: usize = data.len / n_components;

        for (0..n_vecs) |i| {
            for (0..n_components) |d| {
                const datam = data[i * n_components + d];
                try out_stream.print("{any} ", .{datam});
            }

            try out_stream.print("\n", .{});
        }
    }
};
