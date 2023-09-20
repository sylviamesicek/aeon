const std = @import("std");
const meta = std.meta;

/// Checks if a type could be used to describe a system
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

            return info.decls.len == 0 and info.layout == .Auto and info.is_tuple == false and info.backing_integer == null;
        },
        else => return false,
    }
}

pub fn SystemSlice(comptime T: type) type {
    return SystemStruct(T, []f64);
}

pub fn isSystemSlice(comptime T: type) type {
    return isSystemStruct(T, []f64);
}

pub fn SystemSliceConst(comptime T: type) type {
    return SystemStruct(T, []const f64);
}

pub fn isSystemSliceConst(comptime T: type) type {
    return isSystemStruct(T, []const f64);
}

pub fn SystemValue(comptime T: type) type {
    return SystemStruct(T, f64);
}

pub fn isSystemValue(comptime T: type) type {
    return isSystemStruct(T, f64);
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
