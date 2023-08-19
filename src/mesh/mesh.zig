const std = @import("std");
const ArrayList = std.ArrayList;

const geometry = @import("../geometry/geometry.zig");

pub const TileSrc = enum(u2) {
    unchanged,
    /// If this bit is set, it indicates that index points
    /// to a tile on level `l-1` and that the data must be
    /// interpolated to the new level.
    added,
    empty,
};

/// A mapping of from a tile on an old mesh
/// to a new one.
pub const TileMap = packed struct {
    src: TileSrc,
    /// The index of the source tile (if any)
    index: u62,
};

pub fn Mesh(comptime N: usize) type {
    _ = N;
    return struct {

        // Aliases
        const Self = @This();
    };
}
