//! This module defines the concept of a system, ie a coupled set of fields defined over some physical/numerical space.
//! A system can be described with an enum specifying the names of the composite fields.
//!
//! ```
//! const System = enum {
//!     field1,
//!     field2,
//!     // ...
//! };
//! ```
//!
//! This module provides traits for identifying such types, and building new types represents values of these systems at
//! individual cells, slices of the systems in a SoA storage, ect.

const std = @import("std");
const meta = std.meta;

const basis = @import("basis/basis.zig");
const BoundaryCondition = basis.BoundaryCondition;

/// A trait function used for determining if a type desribes a system.
pub fn isSystem(comptime T: type) bool {
    switch (@typeInfo(T)) {
        .Enum => |info| {
            if (info.decls.len == 0 and info.is_exhaustive) {
                return true;
            }
            return false;
        },
        else => return false,
    }
}

/// Builds a struct type with one named field per variant of T, of type F.
pub fn SystemStruct(comptime T: type, comptime F: type) type {
    if (!isSystem(T)) {
        @compileError("T must satisfy isSystem trait");
    }

    const field_infos = meta.fields(T);

    if (field_infos.len == 0) {
        return struct {};
    }

    var struct_fields: [field_infos.len]std.builtin.Type.StructField = undefined;
    var decls = [_]std.builtin.Type.Declaration{};

    inline for (field_infos, 0..) |field, i| {
        struct_fields[i] = .{
            .name = field.name,
            .type = F,
            .default_value = null,
            .is_comptime = false,
            .alignment = @alignOf(F),
        };
    }

    return @Type(.{
        .Struct = .{
            .layout = .Auto,
            .fields = &struct_fields,
            .decls = &decls,
            .is_tuple = false,
        },
    });
}

/// A trait which determines if a type is a valid system struct.
pub fn isSystemStruct(comptime T: type, comptime F: type) bool {
    switch (@typeInfo(T)) {
        .Struct => |info| {
            inline for (info.fields) |field| {
                if (field.type != F) {
                    return false;
                }
            }

            return true;
        },
        else => return false,
    }
}

/// A mutable slice of a system in SoA storage.
pub fn SystemSlice(comptime T: type) type {
    return SystemStruct(T, []f64);
}

pub fn isSystemSlice(comptime T: type) bool {
    return isSystemStruct(T, []f64);
}

/// A constant slice of a system in SoA storage.
pub fn SystemSliceConst(comptime T: type) type {
    return SystemStruct(T, []const f64);
}

pub fn isSystemSliceConst(comptime T: type) bool {
    return isSystemStruct(T, []const f64);
}

/// The value of a system at a point.
pub fn SystemValue(comptime T: type) type {
    return SystemStruct(T, f64);
}

pub fn isSystemValue(comptime T: type) bool {
    return isSystemStruct(T, f64);
}

pub fn SystemBoundaryCondition(comptime T: type) type {
    return SystemStruct(T, BoundaryCondition);
}

pub fn isBoundaryCondition(comptime T: type) bool {
    return isSystemStruct(T, BoundaryCondition);
}

/// Returns the total number of fields in a system
pub fn systemFieldCount(comptime T: type) usize {
    return meta.fields(T).len;
}

/// Iterates the names of fields in a system.
pub fn systemFieldNames(comptime T: type) [systemFieldCount(T)][]const u8 {
    var result: [systemFieldCount(T)][]const u8 = undefined;

    inline for (meta.fields(T), 0..) |field_info, id| {
        result[id] = field_info.name;
    }

    return result;
}

/// Iterates the fields of a struct system.
pub fn systemStructFields(comptime T: type, comptime F: type, sys: SystemStruct(T, F)) [systemFieldCount(T)]F {
    if (!isSystem(T)) {
        @compileError("systemFields only valid for system types");
    }

    var result: [systemFieldCount(T)]F = undefined;

    inline for (meta.fields(T), 0..) |field_info, id| {
        result[id] = @field(sys, field_info.name);
    }

    return result;
}

pub fn systemStructSlice(sys: anytype, offset: usize, total: usize) @TypeOf(sys) {
    var slice: @TypeOf(sys) = undefined;

    inline for (meta.fields(@TypeOf(sys))) |field_info| {
        @field(slice, field_info.name) = @field(sys, field_info.name)[offset..(offset + total)];
    }

    return slice;
}

test "system trait" {
    const expect = std.testing.expect;

    const Sys1 = struct {
        scalar: []const f64,
        other_scalar: []const f64,
        random: f64,
    };

    const Sys2 = enum {
        a,
        b,
        c,
    };

    try expect(!isSystem(Sys1));
    try expect(isSystem(Sys2));
}
