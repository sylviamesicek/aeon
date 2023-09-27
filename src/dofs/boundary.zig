const std = @import("std");
const assert = std.debug.assert;

const basis = @import("../basis/basis.zig");
const geometry = @import("../geometry/geometry.zig");
const system = @import("../system.zig");

const Face = geometry.Face;

const operator = @import("operator.zig");

/// The value of a system at a point.
pub fn SystemBoundaryCondition(comptime T: type) type {
    return system.SystemStruct(T, BoundaryCondition);
}

pub fn isSystemBoundaryCondition(comptime T: type) bool {
    return system.isSystemStruct(T, BoundaryCondition);
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
            if (!(@hasDecl(T, "System") and T.System == type and system.isSystem(T.System))) {
                return false;
            }

            if (!(hasFn("boundary")(T) and T.boundary == fn (T, [N]f64, Face(N)) SystemBoundaryCondition(T.System))) {
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
                            conditions[i] = boundary.boundary(pos, Face{
                                .side = extents[i] > 0,
                                .axis = i,
                            });
                        }
                    }

                    inline for (comptime system.systemFieldNames(T.System)) |name| {
                        // Set the fields of the system to be zero at the target.
                        stencil_space.cellSpace().setValue(target, @field(sys, name), 0.0);

                        var v: f64 = 0.0;
                        var normals: [N]f64 = [1]f64{0.0} ** N;
                        var rhs: f64 = 0.0;

                        for (0..N) |i| {
                            if (extents[i] != 0) {
                                const condition: BoundaryCondition = @field(conditions[i], name);

                                v += condition.value;
                                normals[i] = if (extents[i] > 0) condition.normal else -condition.normal;
                                rhs += condition.rhs;
                            }
                        }

                        var sum: f64 = v * stencil_space.boundaryValue(
                            extents,
                            cell,
                            @field(sys, name),
                        );
                        var coef: f64 = v * stencil_space.boundaryValueCoef(extents);

                        inline for (0..N) |i| {
                            if (extents[i] != 0) {
                                comptime var ranks: [N]usize = [1]usize{0} ** N;
                                ranks[i] = 1;

                                sum += normals[i] * stencil_space.boundaryDerivative(ranks, extents, cell, @field(sys, name));
                                coef += normals[i] * stencil_space.boundaryDerivativeCoef(ranks, extents);
                            }
                        }

                        stencil_space.cellSpace().setValue(target, @field(sys, name), (rhs - sum) / coef);
                    }
                }
            }
        }
    };
}
