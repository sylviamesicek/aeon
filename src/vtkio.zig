const std = @import("std");

pub const VtkCellType = enum {
    quad,
    hexa,

    pub fn dim(self: VtkCellType) i32 {
        return switch (self) {
            .quad => 2,
            .hexa => 3,
        };
    }

    pub fn points(self: VtkCellType) i32 {
        return switch (self) {
            .quad => 4,
            .hexa => 8,
        };
    }
};

pub const VtkDataArray = struct {
    name: []const u8,
    numeric_type: []const u8,
    data: std.ArrayListUnmanaged(f64),
    n_components: i32,

    pub fn write(self: VtkDataArray, out_stream: anytype) @TypeOf(out_stream).Error!void {
        try out_stream.print(
            \\<DataArray type= "{}" Name= "{}" NumberOfComponents="{}" format="ascii"
            \\    {}
            \\</DataArray>
        , .{ self.numeric_type, self.name, self.n_components, self.data });
    }
};

const VtkDataArrayList = std.ArrayListUnmanaged(VtkDataArray);

pub const VtkMesh = struct {
    allocator: std.mem.Allocator,
    cell_type: VtkCellType,
    point_data: VtkDataArrayList,
    cell_data: VtkDataArrayList,

    pub fn init(allocator: std.mem.Allocator, cell_type: VtkCellType) VtkMesh {
        return .{
            .allocator = allocator,
            .cell_type = cell_type,
            .point_data = VtkDataArrayList{},
            .cell_data = VtkDataArrayList{},
        };
    }

    pub fn deinit(self: VtkMesh) void {
        self.point_data.deinit(self.allocator);
        self.cell_data.deinit(self.allocator);
    }

    pub fn add_field(self: VtkMesh, name: []const u8, data: []const f64, dimension: i32) !void {
        return self.point_data.append(self.allocator, .{
            .name = name,
            .numeric_type = "Float64",
            .data = data,
            .n_components = dimension,
        });
    }

    pub fn add_cell_field(self: VtkMesh, name: []const u8, data: []const f64, dimension: i32) !void {
        return self.cell_data.append(self.allocator, .{
            .name = name,
            .numeric_type = "Float64",
            .data = data,
            .n_components = dimension,
        });
    }

    pub fn add_scalar_field(self: VtkMesh, name: []const u8, data: []const f64) !void {
        return self.add_field(name, data, 1);
    }

    pub fn add_cell_scalar_field(self: VtkMesh, name: []const u8, data: []const f64) !void {
        return self.add_cell_field(name, data, 1);
    }

    pub fn add_vector_field(self: VtkMesh, name: []const u8, data: []const f64, dimension: i32) !void {
        return self.add_cell_field(name, data, dimension);
    }

    pub fn add_cell_vector_field(self: VtkMesh, name: []const u8, data: []const f64) !void {
        _ = self;
        _ = data;
        _ = name;
    }

    pub fn write_volume_mesh(self: VtkMesh) void {
        _ = self;
    }

    fn write_header(n_vertices: i32, n_cells: i32, out_stream: anytype) @TypeOf(out_stream).Error!void {
        return out_stream.print(
            \\<VTKFile type="UnstructuedGrid" version="1.0">
            \\<UnstructuredGrid>
            \\<Piece NumberOfPoints="{}" NumberOfCells="{}">
        , .{ n_cells, n_vertices });
    }

    fn write_footer(out_stream: anytype) @TypeOf(out_stream).Error!void {
        return out_stream.print(
            \\</Piece>
            \\</UnstructuredGrid>
            \\</VTKFile>
        , .{});
    }
};
