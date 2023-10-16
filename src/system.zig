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
const enums = std.enums;

const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

pub const EmptySystem = enum {
    pub fn slice() SystemSlice(@This()) {
        return SystemSlice(@This()).view(0, .{});
    }

    pub fn sliceConst() SystemSliceConst(@This()) {
        return SystemSliceConst(@This()).view(0, .{});
    }
};

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

/// The value of a system at a point.
pub fn SystemValue(comptime T: type) type {
    return enums.EnumFieldStruct(T, f64, null);
}

/// A private impl type for generating the SystemSlice
/// and SystemSliceConst types.
fn SystemSliceImpl(comptime T: type, comptime is_const: bool) type {
    const ManyItemPtr: type = if (is_const) [*]const f64 else [*]f64;
    const Slice: type = if (is_const) []const f64 else []f64;
    const Ptrs = enums.EnumFieldStruct(T, ManyItemPtr, null);

    return struct {
        len: usize,
        ptrs: Ptrs,

        const Self = @This();

        pub fn init(allocator: Allocator, len: usize) !Self {
            var ptrs: Ptrs = undefined;

            errdefer {
                inline for (comptime meta.fieldNames(T)) |name| {
                    var slice_to_free: Slice = undefined;
                    slice_to_free.len = len;
                    slice_to_free.ptr = @field(ptrs, name);

                    allocator.free(slice_to_free);
                }
            }

            inline for (comptime meta.fieldNames(T)) |name| {
                const mem = try allocator.alloc(f64, len);
                @field(ptrs, name) = mem.ptr;
            }

            return .{
                .len = len,
                .ptrs = ptrs,
            };
        }

        pub fn deinit(self: Self, allocator: Allocator) void {
            inline for (comptime meta.fieldNames(T)) |name| {
                var slice_to_free: Slice = undefined;
                slice_to_free.len = self.len;
                slice_to_free.ptr = @field(self.ptrs, name);

                allocator.free(slice_to_free);
            }
        }

        pub fn view(len: usize, slices: enums.EnumFieldStruct(T, Slice, null)) Self {
            var ptrs: Ptrs = undefined;

            inline for (comptime meta.fieldNames(T)) |name| {
                assert(@field(slices, name).len == len);
                @field(ptrs, name) = @field(slices, name).ptr;
            }

            return .{
                .len = len,
                .ptrs = ptrs,
            };
        }

        pub fn field(self: Self, comptime sys: T) Slice {
            const ptr = @field(self.ptrs, @tagName(sys));

            var result: Slice = undefined;
            result.len = self.len;
            result.ptr = ptr;

            return result;
        }

        pub fn slice(self: Self, offset: usize, total: usize) Self {
            assert(self.len >= offset + total);

            var ptrs: Ptrs = undefined;

            inline for (comptime meta.fieldNames(T)) |name| {
                const src = @field(self.ptrs, name);

                @field(ptrs, name) = @ptrFromInt(@intFromPtr(src) + offset * @sizeOf(f64));
            }

            return .{
                .len = total,
                .ptrs = ptrs,
            };
        }

        pub fn toConst(self: Self) SystemSliceConst(T) {
            var result: SystemSliceConst(T) = undefined;
            result.len = self.len;

            inline for (comptime meta.fieldNames(T)) |name| {
                @field(result.ptrs, name) = @field(self.ptrs, name);
            }

            return result;
        }

        pub fn assertLen(self: Self, len: usize) void {
            assert(std.meta.fieldNames(T).len == 0 or self.len == len);
        }
    };
}

/// A system slice, ie a memory efficient collection of []f64's for each system,
/// which can be accessed using the `field()` member function.
pub fn SystemSlice(comptime T: type) type {
    return SystemSliceImpl(T, false);
}

/// A system slice const, ie a memory efficient collection of []const f64's for each system,
/// which can be accessed using the `field()` member function.
pub fn SystemSliceConst(comptime T: type) type {
    return SystemSliceImpl(T, true);
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
