const std = @import("std");
const ArrayListUnmanaged = std.ArrayListUnmanaged;
const Allocator = std.mem.Allocator;

/// The type of a cell in an unstructured vtk mesh.
pub const VtkCellType = enum {
    /// A 1d line, consisting of two vertex points.
    line,
    /// A 2d quad, consisting of four vertex points.
    quad,
    /// A 3d hexahedron (distored cube) with 8 vertex points.
    hexa,

    pub fn dimension(self: VtkCellType) usize {
        return switch (self) {
            .line => 1,
            .quad => 2,
            .hexa => 3,
        };
    }

    /// Counts the number of vertices for a given cell type.
    pub fn nvertices(self: VtkCellType) usize {
        return switch (self) {
            .line => 2,
            .quad => 4,
            .hexa => 8,
        };
    }

    /// Returns the VTK spec tag corresponding to the given cell type.
    pub fn tag(self: VtkCellType) i8 {
        return switch (self) {
            .line => 3,
            .quad => 9,
            .hexa => 12,
        };
    }
};

/// A vtk data array backed by an array of doubles.
const VtkDataArray = struct {
    name: []const u8,
    data: []const f64,
    n_components: usize,

    fn deinit(self: *VtkDataArray, allocator: Allocator) void {
        self.data.deinit(allocator);
    }
};

const VtkDataArrayList = ArrayListUnmanaged(VtkDataArray);

/// Stores data for a vtk (unstructured) grid. This data structure does not own the fields or point data passed
/// to it, only references.
pub const VtuMeshOutput = struct {
    /// A stored allocator for internal `ArrayList`s.
    allocator: Allocator,
    /// Stores data which has been associated with each point.
    point_data: VtkDataArrayList,
    /// Stores data which has been associated with each cell.
    cell_data: VtkDataArrayList,
    /// Stores the type of cell that make up this grid (currently must be global).
    cell_type: VtkCellType,
    /// Stores the positions of each point in the grid.
    points: []const f64,
    /// Stores the vertices of each cell in the grid.
    vertices: []const usize,

    const Self = @This();

    /// Global config of an unstructured grid. Describes cell type, point positions, and cell vertices.
    pub const Config = struct {
        cell_type: VtkCellType,
        points: []const f64,
        vertices: []const usize,
    };

    /// Initialises a new unstructured grid using an allocator and a config.
    pub fn init(allocator: Allocator, config: Config) !Self {
        return .{
            .allocator = allocator,
            .point_data = VtkDataArrayList{},
            .cell_data = VtkDataArrayList{},
            .cell_type = config.cell_type,
            .points = config.points,
            .vertices = config.vertices,
        };
    }

    /// Deinitalises an unstructured grid, freeing all data.
    pub fn deinit(self: *Self) void {
        self.point_data.deinit(self.allocator);
        self.cell_data.deinit(self.allocator);
    }

    /// Adds a data field associated with each point to the vtk unstructured grid.
    pub fn addField(self: *Self, name: []const u8, data: []const f64, dimension: usize) !void {
        return self.point_data.append(self.allocator, .{
            .name = name,
            .data = data,
            .n_components = dimension,
        });
    }

    /// Adds a data field associated with each cell to the vtk unstructured grid.
    pub fn addCellField(self: *Self, name: []const u8, data: []const f64, dimension: usize) !void {
        return self.cell_data.append(self.allocator, .{
            .name = name,
            .data = data,
            .n_components = dimension,
        });
    }

    /// Writes the unstructured grid into an output stream in .vtu format.
    pub fn write(self: Self, out_stream: anytype) @TypeOf(out_stream).Error!void {
        const dimension = self.cell_type.dimension();
        const n_points: usize = self.points.len / dimension;

        const n_vertices = self.cell_type.nvertices();
        const n_cells: usize = self.vertices.len / n_vertices;

        try writeHeader(n_points, n_cells, out_stream);
        try writePoints(dimension, self.points, out_stream);
        try writeCells(self.cell_type, self.vertices, out_stream);
        try self.writePointData(out_stream);
        try self.writeCellData(out_stream);
        try writeFooter(out_stream);
    }

    fn writePointData(self: Self, out_stream: anytype) @TypeOf(out_stream).Error!void {
        try out_stream.print("<PointData>\n", .{});
        for (self.point_data.items) |point_data| {
            try writeDataArray(point_data, out_stream);
        }
        try out_stream.print("</PointData>\n", .{});
    }

    fn writeCellData(self: Self, out_stream: anytype) @TypeOf(out_stream).Error!void {
        try out_stream.print("<CellData>\n", .{});
        for (self.cell_data.items) |cell_data| {
            try writeDataArray(cell_data, out_stream);
        }
        try out_stream.print("</CellData>\n", .{});
    }

    fn writeHeader(n_points: usize, n_cells: usize, out_stream: anytype) @TypeOf(out_stream).Error!void {
        return out_stream.print(
            \\<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian">
            \\<UnstructuredGrid>
            \\<Piece NumberOfPoints="{}" NumberOfCells="{}">
            \\
        , .{ n_points, n_cells });
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
            \\<DataArray type="Float64" NumberOfComponents="3" format="ascii">
            \\
        , .{});

        const n_vecs: usize = points.len / n_components;

        for (0..n_vecs) |i| {
            for (0..n_components) |d| {
                const datam: f64 = points[i * n_components + d];
                try out_stream.print("{} ", .{datam});
            }

            for (n_components..3) |_| {
                try out_stream.print("0 ", .{});
            }

            try out_stream.print("\n", .{});
        }

        try out_stream.print(
            \\</DataArray>
            \\</Points>
            \\
        , .{});
    }

    fn writeCells(cell_type: VtkCellType, cells: []const usize, out_stream: anytype) @TypeOf(out_stream).Error!void {
        const n_vertices = cell_type.nvertices();
        const n_cells: usize = cells.len / n_vertices;
        const tag = cell_type.tag();

        // Connectivity Info
        try out_stream.print(
            \\<Cells>
            \\<DataArray type="Int64" Name="connectivity" format="ascii">
            \\
        , .{});

        try writeVecArray(usize, n_vertices, cells, out_stream);

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

        try out_stream.print("</DataArray>\n", .{});

        // Tags
        try out_stream.print(
            \\<DataArray type="Int64" Name="types" format="ascii" RangeMin="{}" RangeMax="{}">
            \\
        , .{ tag, tag });

        for (0..n_cells) |_| {
            try out_stream.print("{}\n", .{tag});
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
        try writeVecArray(f64, array.n_components, array.data, out_stream);
        try out_stream.print("</DataArray>\n", .{});
    }

    fn writeVecArray(comptime T: type, n_components: usize, data: []const T, out_stream: anytype) @TypeOf(out_stream).Error!void {
        const n_vecs: usize = data.len / n_components;

        for (0..n_vecs) |i| {
            for (0..n_components) |d| {
                const datam: T = data[i * n_components + d];
                try out_stream.print("{any} ", .{datam});
            }

            try out_stream.print("\n", .{});
        }
    }
};
