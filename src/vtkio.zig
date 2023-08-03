const std = @import("std");

const VtkDataNode = struct {
    name: []const u8,
    numeric_type: []const u8,
    data: std.ArrayList(f64),
    n_components: i32,

    pub fn stringify(self: VtkDataNode, out_stream: anytype) @TypeOf(out_stream).Error!void {
        try out_stream.print(
            \\<DataArray type= "{}" Name= "{}" NumberOfComponents="{}" format="ascii"
            \\    {}
            \\</DataArray>
        , .{ self.numeric_type, self.name, self.n_components, self.data });
    }
};

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

const VtuData = struct {
    allocator: std.mem.Allocator,
    cell_type: VtkCellType,

    pub fn init(allocator: std.mem.Allocator, cell_type: VtkCellType) VtuData {
        return .{
            .allocator = allocator,
            .cell_type = cell_type,
        };
    }

    pub fn deinit(self: VtuData) void {
        _ = self;
    }

    pub fn add_field(self: VtuData, name: []const u8, data: []const f64, dimension: i32) !void {
        _ = self;
        _ = dimension;
        _ = data;
        _ = name;
    }

    pub fn add_cell_field(self: VtuData, name: []const u8, data: []const f64, dimension: i32) !void {
        _ = self;
        _ = dimension;
        _ = data;
        _ = name;
    }

    pub fn add_scalar_field(self: VtuData, name: []const u8, data: []const f64) !void {
        _ = self;
        _ = data;
        _ = name;
    }

    pub fn add_cell_scalar_field(self: VtuData, name: []const u8, data: []const f64) !void {
        _ = self;
        _ = data;
        _ = name;
    }

    pub fn add_vector_field(self: VtuData, name: []const u8, data: []const f64) !void {
        _ = self;
        _ = data;
        _ = name;
    }

    pub fn add_cell_vector_field(self: VtuData, name: []const u8, data: []const f64) !void {
        _ = self;
        _ = data;
        _ = name;
    }

    pub fn write_volume_mesh(self: VtuData) void {
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

    fn write_data_array(name: []const u8, out_stream: anytype) @TypeOf(out_stream).Error!void {
        _ = name;
        return out_stream.print("", .{});
    }
};
