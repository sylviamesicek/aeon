const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const system = @import("../system.zig");

/// A cache of memory for a system, commonly used to allocate scratch space for ghost cells
/// of a system during elliptic or hyperbolic solving.
pub fn SystemChunk(comptime System: type) type {
    return struct {
        allocator: Allocator,
        len: usize,
        sys: system.SystemSlice(System),

        const Self = @This();

        pub fn init(allocator: Allocator, len: usize) !Self {
            assert(len > 0);

            var sys: system.SystemSlice(System) = undefined;

            inline for (comptime system.systemFieldNames(System)) |name| {
                @field(sys, name) = &.{};
            }

            errdefer {
                inline for (comptime system.systemFieldNames(System)) |name| {
                    allocator.free(@field(sys, name));
                }
            }

            inline for (comptime system.systemFieldNames(System)) |name| {
                @field(sys, name) = try allocator.alloc(f64, len);
            }

            return .{
                .allocator = allocator,
                .len = len,
                .sys = sys,
            };
        }

        pub fn deinit(self: *Self) void {
            inline for (comptime system.systemFieldNames(System)) |name| {
                self.allocator.free(@field(self.sys, name));
            }
        }

        pub fn slice(self: Self, total: usize) system.SystemSlice(System) {
            assert(total <= self.len);

            var result: system.SystemSlice(System) = undefined;

            inline for (comptime system.systemFieldNames(System)) |name| {
                @field(result, name) = @field(self.sys, name)[0..total];
            }

            return result;
        }

        pub fn sliceConst(self: Self, total: usize) system.SystemSliceConst(System) {
            assert(total <= self.len);

            var result: system.SystemSliceConst(System) = undefined;

            inline for (comptime system.systemFieldNames(System)) |name| {
                @field(result, name) = @field(self.sys, name)[0..total];
            }

            return result;
        }
    };
}
