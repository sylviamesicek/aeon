const std = @import("std");
const assert = std.debug.assert;

const basis = @import("../basis/basis.zig");
const geometry = @import("../geometry/geometry.zig");
const system = @import("../system.zig");

const Face = geometry.Face;

/// The value of a system at a point.
pub fn SystemBoundaryCondition(comptime T: type) type {
    return std.enums.EnumFieldStruct(T, BoundaryCondition, null);
}

/// Represents a boundary condition as returned by a BoundaryOperator. Specifies a robin boundary condition along each
/// face as a function of position.
pub const BoundaryCondition = struct {
    value: f64,
    normal: f64,
    rhs: f64,

    const Self = @This();

    /// Constructs a diritchlet boundary condition.
    pub fn diritchlet(rhs: f64) Self {
        return .{
            .value = 1.0,
            .normal = 0.0,
            .rhs = rhs,
        };
    }

    /// Constructs a nuemann boundary condition.
    pub fn nuemann(rhs: f64) Self {
        return .{
            .value = 0.0,
            .normal = 1.0,
            .rhs = rhs,
        };
    }

    /// Constructs a robin boundary condition.
    pub fn robin(value: f64, normal: f64, rhs: f64) Self {
        return .{
            .value = value,
            .normal = normal,
            .rhs = rhs,
        };
    }
};

pub fn isSystemBoundary(comptime N: usize) fn (comptime T: type) bool {
    const hasFn = std.meta.trait.hasFn;

    const Closure = struct {
        fn trait(comptime T: type) bool {
            if (comptime !(@hasDecl(T, "System") and @TypeOf(T.System) == type and system.isSystem(T.System))) {
                return false;
            }

            if (!(hasFn("boundary")(T) and @TypeOf(T.boundary) == fn (T, [N]f64, Face(N)) SystemBoundaryCondition(T.System))) {
                return false;
            }

            return true;
        }
    };

    return Closure.trait;
}

pub fn BoundaryUtils(comptime N: usize, comptime O: usize) type {
    return struct {
        const StencilSpace = basis.StencilSpace(N, O);
        const Region = geometry.Region(N);

        pub fn fillBoundary(
            comptime E: usize,
            stencil_space: StencilSpace,
            boundary: anytype,
            sys: system.SystemSlice(@TypeOf(boundary).System),
        ) void {
            if (comptime !isSystemBoundary(N)(@TypeOf(boundary))) {
                @compileError("Boundary must satisfy isSystemBoundary trait.");
            }

            const regions = comptime Region.orderedRegions();

            inline for (comptime regions[1..]) |region| {
                fillBoundaryRegion(E, region, stencil_space, boundary, sys);
            }
        }

        pub fn fillBoundaryRegion(
            comptime E: usize,
            comptime region: Region,
            stencil_space: StencilSpace,
            boundary: anytype,
            sys: system.SystemSlice(@TypeOf(boundary).System),
        ) void {
            const L = 2 * O + 2;
            // const L = 1;

            const T = @TypeOf(boundary);

            if (comptime !isSystemBoundary(N)(T)) {
                @compileError("Boundary must satisfy isSystemBoundary trait.");
            }

            var inner_face_cells = region.innerFaceIndices(stencil_space.size);

            while (inner_face_cells.next()) |cell| {
                comptime var extent_indices = region.extentOffsets(E);

                inline while (comptime extent_indices.next()) |extents| {

                    // Compute target cell for the given extent indices
                    var target: [N]isize = undefined;

                    inline for (0..N) |i| {
                        target[i] = cell[i] + extents[i];
                    }

                    // Compute position of boundary given extents
                    const pos: [N]f64 = stencil_space.boundaryPosition(extents, cell);

                    // Compute conditions for the given field.
                    var conditions: [N]SystemBoundaryCondition(T.System) = undefined;

                    inline for (0..N) |i| {
                        if (extents[i] != 0) {
                            conditions[i] = boundary.boundary(pos, Face(N){
                                .side = extents[i] > 0,
                                .axis = i,
                            });
                        }
                    }

                    inline for (comptime std.enums.values(T.System)) |field| {
                        // Set the fields of the system to be zero at the target.
                        stencil_space.nodeSpace().setValue(target, sys.field(field), 0.0);

                        var v: f64 = 0.0;
                        var normals: [N]f64 = [1]f64{0.0} ** N;
                        var rhs: f64 = 0.0;

                        for (0..N) |i| {
                            if (extents[i] != 0) {
                                const condition: BoundaryCondition = @field(conditions[i], @tagName(field));

                                v += condition.value;
                                normals[i] = condition.normal;
                                rhs += condition.rhs;
                            }
                        }

                        var sum: f64 = v * stencil_space.boundaryValue(
                            L,
                            extents,
                            cell,
                            sys.field(field),
                        );
                        var coef: f64 = v * stencil_space.boundaryValueCoef(L, extents);

                        inline for (0..N) |i| {
                            if (extents[i] != 0) {
                                comptime var ranks: [N]usize = [1]usize{0} ** N;
                                ranks[i] = 1;

                                sum += normals[i] * stencil_space.boundaryDerivative(L, extents, ranks, cell, sys.field(field));
                                coef += normals[i] * stencil_space.boundaryDerivativeCoef(L, extents, ranks);
                            }
                        }

                        stencil_space.nodeSpace().setValue(target, sys.field(field), (rhs - sum) / coef);
                    }
                }
            }
        }
    };
}

test "boundary filling" {
    const TestSystem = enum {
        func,
    };

    const Diritchlet = struct {
        pub const System = TestSystem;

        pub fn boundary(self: @This(), pos: [2]f64, face: Face(2)) SystemBoundaryCondition(System) {
            _ = face;
            _ = pos;
            _ = self;
            return .{
                .func = BoundaryCondition.diritchlet(0.0),
            };
        }
    };

    const math = std.math;
    const allocator = std.testing.allocator;

    const a = 0.0;
    const b = 2.0 * math.pi;

    const total_cells = 32;

    const stencil_space = basis.StencilSpace(2, 2){
        .physical_bounds = .{
            .origin = [1]f64{a} ** 2,
            .size = [1]f64{b - a} ** 2,
        },
        .size = [1]usize{total_cells} ** 2,
    };
    const cell_space = stencil_space.nodeSpace();

    var func = try system.SystemSlice(TestSystem).init(allocator, cell_space.total());
    defer func.deinit(allocator);

    var cells = cell_space.nodes(0);

    while (cells.next()) |cell| {
        const pos = stencil_space.position(cell);
        cell_space.setValue(cell, func.field(.func), math.sin(pos[0]) * math.sin(pos[1]));
    }

    BoundaryUtils(2, 2).fillBoundary(2, stencil_space, Diritchlet{}, func);

    // for (0..4) |i| {
    //     for (0..4) |j| {
    //         const is: isize = @intCast(i);
    //         const js: isize = @intCast(j);

    //         const cell = [2]isize{ is - 2, js - 2 };
    //         std.debug.print("Cell {any}, value {}\n", .{ cell, cell_space.value(cell, func.field(.func)) });
    //     }
    // }

    const ext = cell_space.value([2]isize{ -2, -2 }, func.field(.func));
    _ = ext;
    const int = cell_space.value([2]isize{ 1, 1 }, func.field(.func));
    _ = int;

    // std.debug.print("Exterior {} Interior {}", .{ ext, int });
}
