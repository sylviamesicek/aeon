const std = @import("std");
const meta = std.meta;

/// The type of the slice in a system, whether []const f64 or []f64
pub fn SystemSliceType(comptime T: type) type {
    if (isConstSystem(T)) {
        return []const f64;
    } else if (isMutableSystem(T)) {
        return []f64;
    } else {
        @compileError("SystemSliceType only valid for system types");
    }
}

/// Returns an enum representing each field in the system.
pub fn SystemFieldEnum(comptime T: type) type {
    return meta.FieldEnum(T);
}

pub fn SystemValueStruct(comptime T: type) type {
    if (!isSystem(T)) {
        @compileError("SystemValueStruct may only be called on system types.");
    }

    comptime var fields: [systemFieldCount(T)]std.builtin.Type.StructField = undefined;

    for (systemFieldNames(T), 0..) |name, id| {
        fields[id] = .{
            .name = name,
            .type = f64,
            .default_value = null,
            .is_comptime = false,
            .alignment = @alignOf(T),
        };
    }

    return @Type(.{ .Struct = .{
        .layout = .Auto,
        .fields = &fields,
        .decls = &.{},
        .is_tuple = false,
    } });
}

// pub fn SystemUnion(comptime T: type, comptime U: type) type {
//     // Determine field type
//     comptime var FieldType: type = undefined;

//     if (isConstSystem(T) and isConstSystem(U)) {
//         FieldType = []const f64;
//     } else if (isMutableSystem(T) and isMutableSystem(U)) {
//         FieldType = []f64;
//     } else {
//         @compileError("SystemUnion can only operate on two system which are both of the same constness.");
//     }

//     comptime var fields: [systemFieldCount(T) + systemFieldCount(U)]std.builtin.Type.StructField = undefined;
//     comptime var index = 0;

//     for (systemFieldNames(T)) |name| {
//         fields[index] = .{
//             .name = name,
//             .type = FieldType,
//             .default_value = null,
//             .is_comptime = false,
//             .alignment = @alignOf(FieldType),
//         };
//         index += 1;
//     }

//     for (systemFieldNames(U)) |name| {
//         fields[index] = .{
//             .name = name,
//             .type = FieldType,
//             .default_value = null,
//             .is_comptime = false,
//             .alignment = @alignOf(FieldType),
//         };
//         index += 1;
//     }

//     return @Type(.{ .Struct = .{
//         .layout = .Auto,
//         .fields = &fields,
//         .decls = &.{},
//         .is_tuple = false,
//     } });
// }

/// Returns the total number of fields in a system
pub fn systemFieldCount(comptime T: type) usize {
    return meta.fields(T).len;
}

/// Iterates the names of fields in a system.
pub fn systemFieldNames(comptime T: type) [systemFieldCount(T)][:0]const u8 {
    if (!isSystem(T)) {
        @compileError("systemFieldNames only valid for system types");
    }

    var result: [systemFieldCount(T)][:0]const u8 = undefined;

    inline for (meta.fields(T), 0..) |field_info, id| {
        result[id] = field_info.name;
    }

    return result;
}

/// Iterates the fields of a system.
pub fn systemFields(comptime T: type, sys: T) [systemFieldCount(T)]SystemSliceType(T) {
    if (!isSystem(T)) {
        @compileError("systemFields only valid for system types");
    }

    var result: [systemFieldCount(T)]SystemSliceType(T) = undefined;

    inline for (meta.fields(T), 0..) |field_info, id| {
        result[id] = @field(sys, field_info.name);
    }

    return result;
}

/// Get a field of a system given a value from its field enum.
pub fn systemField(sys: anytype, comptime field: SystemFieldEnum(@TypeOf(sys))) SystemSliceType(@TypeOf(sys)) {
    return @field(sys, @tagName(field));
}

pub fn systemSlice(sys: anytype, offset: usize, total: usize) @TypeOf(sys) {
    const T = @TypeOf(sys);

    if (!isSystem(T)) {
        @compileError("systemSlice() may only be called on valid systems");
    }

    var slice: T = undefined;

    inline for (meta.fields(T)) |field_info| {
        @field(slice, field_info.name) = @field(sys, field_info.name)[offset..(offset + total)];
    }

    return slice;
}

/// A trait function to determine whether a
/// type is a system. Aka a struct where all fields
/// are const f64 slices.
pub fn isConstSystem(comptime T: type) bool {
    switch (@typeInfo(T)) {
        .Struct => |info| {
            inline for (info.fields) |field| {
                if (field.type != []const f64) {
                    return false;
                }
            }

            return true;
        },
        else => return false,
    }
}

/// A trait function to determine whether a
/// type is a system. Aka a struct where all fields
/// are mutable f64 slices.
pub fn isMutableSystem(comptime T: type) bool {
    switch (@typeInfo(T)) {
        .Struct => |info| {
            inline for (info.fields) |field| {
                if (field.type != []f64) {
                    return false;
                }
            }

            return true;
        },
        else => return false,
    }
}

/// An alias for `isConstSystem(T) or isMutableSystem(T)`
pub fn isSystem(comptime T: type) bool {
    return isConstSystem(T) or isMutableSystem(T);
}

test "system trait" {
    const expect = std.testing.expect;

    const Sys1 = struct {
        scalar: []const f64,
        other_scalar: []const f64,
        random: f64,
    };

    const Sys2 = struct {
        scalar: []f64,
        other_scalar: []f64,
        random: []f64,
    };

    try expect(!isConstSystem(Sys1));
    try expect(isMutableSystem(Sys2));
    try expect(SystemSliceType(Sys2) == []f64);
    try expect(systemFieldCount(Sys2) == 3);
}
