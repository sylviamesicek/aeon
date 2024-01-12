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
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

pub fn System(comptime Tag: type) type {
    return SystemImpl(Tag, false);
}

pub fn SystemConst(comptime Tag: type) type {
    return SystemImpl(Tag, true);
}

pub fn isSystemTag(comptime T: type) bool {
    switch (@typeInfo(T)) {
        .Enum => |info| {
            if (info.is_exhaustive) {
                return true;
            }
            return false;
        },
        else => return false,
    }
}

fn SystemImpl(comptime Tag: type, comptime is_const: bool) type {
    const ManyItemPtr: type = if (is_const) [*]const f64 else [*]f64;
    const Slice: type = if (is_const) []const f64 else []f64;
    const Slices = std.enums.EnumFieldStruct(Tag, Slice, null);
    const Ptrs = std.enums.EnumFieldStruct(Tag, ManyItemPtr, null);

    const fields = std.meta.fieldNames(Tag);

    return struct {
        len: usize,
        ptrs: Ptrs,

        pub fn init(allocator: Allocator, len: usize) !@This() {
            var slices: Slices = undefined;

            inline for (fields) |name| {
                @field(slices, name) = &.{};
            }

            errdefer {
                inline for (fields) |name| {
                    allocator.free(@field(slices, name));
                }
            }

            inline for (fields) |name| {
                @field(slices, name) = try allocator.alloc(f64, len);
            }

            return view(len, slices);
        }

        pub fn deinit(self: @This(), allocator: Allocator) void {
            inline for (fields) |name| {
                var slice_to_free: Slice = undefined;
                slice_to_free.len = self.len;
                slice_to_free.ptr = @field(self.ptrs, name);

                allocator.free(slice_to_free);
            }
        }

        pub fn view(len: usize, slices: Slices) @This() {
            var ptrs: Ptrs = undefined;

            inline for (fields) |name| {
                assert(@field(slices, name).len == len);
                @field(ptrs, name) = @field(slices, name).ptr;
            }

            return .{
                .len = len,
                .ptrs = ptrs,
            };
        }

        pub fn field(self: @This(), comptime sys: Tag) Slice {
            const ptr = @field(self.ptrs, @tagName(sys));

            var result: Slice = undefined;
            result.len = self.len;
            result.ptr = ptr;

            return result;
        }

        pub fn slice(self: @This(), offset: usize, total: usize) @This() {
            assert(self.len >= offset + total);

            var ptrs: Ptrs = undefined;

            inline for (fields) |name| {
                const src = @field(self.ptrs, name);

                @field(ptrs, name) = @ptrFromInt(@intFromPtr(src) + offset * @sizeOf(f64));
            }

            return .{
                .len = total,
                .ptrs = ptrs,
            };
        }

        pub fn toConst(self: @This()) SystemImpl(Tag, true) {
            var result: SystemImpl(Tag, true) = undefined;
            result.len = self.len;

            inline for (fields) |name| {
                @field(result.ptrs, name) = @field(self.ptrs, name);
            }

            return result;
        }
    };
}

test "system tags" {
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

    try expect(!isSystemTag(Sys1));
    try expect(isSystemTag(Sys2));
}
