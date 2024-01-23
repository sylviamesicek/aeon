const std = @import("std");
const ArrayListUnmanaged = std.ArrayListUnmanaged(usize);
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

pub const Range = struct {
    start: usize,
    end: usize,
};

/// A map from elements to ranges.
pub const RangeMap = struct {
    offsets: ArrayListUnmanaged = .{},

    pub fn deinit(self: *@This(), allocator: Allocator) void {
        self.offsets.deinit(allocator);
    }

    pub fn offset(self: @This(), idx: usize) usize {
        return self.offsets.items[idx];
    }

    pub fn size(self: @This(), idx: usize) usize {
        return self.offsets.items[idx + 1] - self.offsets.items[idx];
    }

    pub fn range(self: @This(), idx: usize) Range {
        return .{
            .start = self.offset(idx),
            .end = self.offset(idx + 1),
        };
    }

    pub fn slice(self: @This(), idx: usize, data: anytype) @TypeOf(data) {
        const r = self.range(idx);
        return data[r.start..r.end];
    }

    pub fn total(self: @This()) usize {
        return self.offsets.items[self.offsets.items.len - 1];
    }

    pub fn len(self: @This()) usize {
        return self.offsets.items.len;
    }

    pub fn set(self: @This(), idx: usize, val: usize) void {
        self.offsets.items[idx] = val;
    }

    pub fn resize(self: *@This(), allocator: Allocator, l: usize) !void {
        return self.offsets.resize(allocator, l);
    }

    pub fn clear(self: *@This()) void {
        self.offsets.clearRetainingCapacity();
    }

    pub fn append(self: *@This(), allocator: Allocator, val: usize) !void {
        return self.offsets.append(allocator, val);
    }
};
