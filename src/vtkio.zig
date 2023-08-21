const std = @import("std");
const ArrayListUnmanaged = std.ArrayListUnmanaged;

/// The type of a cell in an unstructured vtk mesh.
pub const VtkCellType = enum {
    /// A 2d quad, consisting of four vertex points.
    quad,
    /// A 3d hexahedron (distored cube) with 8 vertex points.
    hexa,

    pub fn dimension(self: VtkCellType) usize {
        return switch (self) {
            .quad => 2,
            .hexa => 3,
        };
    }

    /// Counts the number of vertices for a given cell type.
    pub fn n_vertices(self: VtkCellType) usize {
        return switch (self) {
            .quad => 4,
            .hexa => 8,
        };
    }

    /// Returns the VTK spec tag corresponding to the given cell type.
    pub fn tag(self: VtkCellType) i8 {
        return switch (self) {
            .quad => 9,
            .hexa => 12,
        };
    }
};

/// A vtk data array backed by an array of doubles.
const VtkDataArray = struct {
    name: []const u8,
    data: ArrayListUnmanaged(f64),
    n_components: usize,
};

const VtkDataArrayList = ArrayListUnmanaged(VtkDataArray);

/// Stores data for a vtk (unstructured) grid.
pub const VtkUnstructuredGrid = struct {
    /// A stored allocator for internal `ArrayList`s.
    allocator: std.mem.Allocator,
    /// Stores data which has been associated with each point.
    point_data: VtkDataArrayList,
    /// Stores data which has been associated with each cell.
    cell_data: VtkDataArrayList,
    /// Stores the type of cell that make up this grid (currently must be global).
    cell_type: VtkCellType,
    /// Stores the positions of each point in the grid.
    points: []f64,
    /// Stores the vertices of each cell in the grid.
    vertices: []i64,

    /// Global config of an unstructured grid. Describes cell type, point positions, and cell vertices.
    pub const Config = struct {
        cell_type: VtkCellType,
        points: []const f64,
        vertices: []const i64,
    };

    /// Initialises a new unstructured grid using an allocator and a config.
    pub fn init(allocator: std.mem.Allocator, config: Config) !VtkUnstructuredGrid {
        var points_owned: std.ArrayListUnmanaged(f64) = try std.ArrayListUnmanaged(f64).initCapacity(allocator, config.points.len);
        errdefer points_owned.deinit(allocator);

        points_owned.appendSliceAssumeCapacity(config.points);

        var vertices_owned: std.ArrayListUnmanaged(i64) = try std.ArrayListUnmanaged(i64).initCapacity(allocator, config.vertices.len);
        errdefer vertices_owned.deinit(allocator);

        vertices_owned.appendSliceAssumeCapacity(config.vertices);

        var points = try allocator.alloc(f64, config.points.len);
        errdefer allocator.free(points);

        @memcpy(points, config.points);

        var vertices = try allocator.alloc(i64, config.vertices.len);
        errdefer allocator.free(vertices);

        @memcpy(vertices, config.vertices);

        return .{
            .allocator = allocator,
            .point_data = VtkDataArrayList{},
            .cell_data = VtkDataArrayList{},
            .cell_type = config.cell_type,
            .points = points,
            .vertices = vertices,
        };
    }

    /// Deinitalises an unstructured grid, freeing all data.
    pub fn deinit(self: *VtkUnstructuredGrid) void {
        self.allocator.free(self.point_data);
        self.allocator.free(self.cell_data);
    }

    /// Adds a data field associated with each point to the vtk unstructured grid.
    pub fn addField(self: *VtkUnstructuredGrid, name: []const u8, data: []const f64, dimension: i32) !void {
        return self.point_data.append(self.allocator, .{
            .name = name,
            .data = data,
            .n_components = dimension,
        });
    }

    /// Adds a data field associated with each cell to the vtk unstructured grid.
    pub fn addCellField(self: *VtkUnstructuredGrid, name: []const u8, data: []const f64, dimension: i32) !void {
        return self.cell_data.append(self.allocator, .{
            .name = name,
            .data = data,
            .n_components = dimension,
        });
    }

    /// Writes the unstructured grid into an output stream in .vtu format.
    pub fn write(self: VtkUnstructuredGrid, out_stream: anytype) @TypeOf(out_stream).Error!void {
        const dimension = self.cell_type.dimension();
        const n_points: usize = self.points.len / dimension;

        const n_vertices = self.cell_type.n_vertices();
        const n_cells: usize = self.vertices.len / n_vertices;

        try writeHeader(n_points, n_cells, out_stream);
        try writePoints(dimension, self.points, out_stream);
        try writeCells(self.cell_type, self.vertices, out_stream);
        try self.writePointData(out_stream);
        try self.writeCellData(out_stream);
        try writeFooter(out_stream);
    }

    fn writePointData(self: VtkUnstructuredGrid, out_stream: anytype) @TypeOf(out_stream).Error!void {
        try out_stream.print("<PointData>\n", .{});
        for (self.point_data.items) |point_data| {
            try writeDataArray(point_data, out_stream);
        }
        try out_stream.print("</PointData>\n", .{});
    }

    fn writeCellData(self: VtkUnstructuredGrid, out_stream: anytype) @TypeOf(out_stream).Error!void {
        try out_stream.print("<CellData>\n", .{});
        for (self.point_data.items) |point_data| {
            try writeDataArray(point_data, out_stream);
        }
        try out_stream.print("</CellData>\n", .{});
    }

    fn writeHeader(n_points: usize, n_cells: usize, out_stream: anytype) @TypeOf(out_stream).Error!void {
        return out_stream.print(
            \\<VTKFile type="UnstructuredGrid" version="1.0">
            \\<UnstructuredGrid>
            \\<Piece NumberOfPoints="{}" NumberOfCells="{}">
            \\
        , .{ n_cells, n_points });
    }

    fn writeFooter(out_stream: anytype) @TypeOf(out_stream).Error!void {
        return out_stream.print(
            \\</Piece>
            \\</UnstructuredGrid>
            \\</VTKFile>
            \\
        , .{});
    }

    fn writePoints(n_components: usize, points: []const f64, out_stream: anytype) @TypeOf(out_stream).Error!void {
        try out_stream.print(
            \\<Points>
            \\<DataArray type="Float64" NumberOfComponents="{}" format="ascii">
            \\
        , .{n_components});

        try writeVecArray(f64, n_components, points, out_stream);

        try out_stream.print(
            \\</DataArray>
            \\</Points>
            \\
        , .{});
    }

    fn writeCells(cell_type: VtkCellType, cells: []const i64, out_stream: anytype) @TypeOf(out_stream).Error!void {
        const n_vertices = cell_type.n_vertices();
        const tag = cell_type.tag();

        // Connectivity Info
        try out_stream.print(
            \\<Cells>
            \\<DataArray type="Int64" Name="connectivity" NumberOfComponents="{}" format="ascii">
            \\
        , .{n_vertices});

        try writeVecArray(i64, n_vertices, cells, out_stream);

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

    fn writeDataArray(array: VtkDataArray, out_stream: anytype) @TypeOf(out_stream).Error!void {
        try out_stream.print(
            \\<DataArray type="Float64" Name="{s}" NumberOfComponents="{}" format="ascii">
            \\
        , .{ array.name, array.n_components });
        try writeVecArray(f64, array.n_components, array.data.items, out_stream);
        try out_stream.print("</DataArray>\n", .{});
    }

    fn writeVecArray(comptime T: type, n_components: usize, data: []const T, out_stream: anytype) @TypeOf(out_stream).Error!void {
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
