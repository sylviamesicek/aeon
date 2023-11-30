//! Provides method-agnostic types and routines for various kinds of meshes (block structured, quadtree based, etc.)

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

// Submodules

const engine = @import("engine.zig");

pub const Engine = engine.Engine;
pub const isOperator = engine.isOperator;

// ***************************
// Offset Maps ***************
// ***************************

/// A map from blocks (defined in a mesh agnostic way) into `CellVector`s. This is filled by the appropriate mesh routine.
pub const CellMap = struct {
    offsets: ArrayList(usize) = .{},

    pub fn init(allocator: Allocator) CellMap {
        const offsets = ArrayList(usize).init(allocator);
        return .{ .offsets = offsets };
    }

    /// Frees a `CellMap`.
    pub fn deinit(self: *CellMap) void {
        self.offsets.deinit();
    }

    /// Returns the cells offset for a given block.
    pub fn offset(self: CellMap, block: usize) usize {
        return self.offsets.items[block];
    }

    /// Returns the total number of cells in a given block.
    pub fn total(self: CellMap, block: usize) usize {
        return self.offsets.items[block + 1] - self.offsets.items[block];
    }

    /// Returns of slice of cells for a given block.
    pub fn slice(self: CellMap, block: usize, data: anytype) @TypeOf(data) {
        return data[self.offsets.items[block]..self.offsets.items[block + 1]];
    }

    /// Returns the number of cells in the total map.
    pub fn numCells(self: CellMap) usize {
        return self.offsets.items[self.offsets.len - 1];
    }
};

/// A map from patches (defined in a mesh agnostic way) into tile vectors. For example, tags for mesh refinement
/// are most often stored in tile vectors. This is filled by the appropriate mesh routine.
pub const TileMap = struct {
    offsets: ArrayList(usize) = .{},

    pub fn init(allocator: Allocator) TileMap {
        const offsets = ArrayList(usize).init(allocator);
        return .{ .offsets = offsets };
    }

    /// Frees a `NodeMap`.
    pub fn deinit(self: *TileMap) void {
        self.offsets.deinit();
    }

    /// Returns the node offset for a given block.
    pub fn offset(self: TileMap, patch: usize) usize {
        return self.offsets.items[patch];
    }

    /// Returns the total number of nodes in a given block.
    pub fn total(self: TileMap, patch: usize) usize {
        return self.offsets.items[patch + 1] - self.offsets.items[patch];
    }

    /// Returns of slice of nodes for a given block.
    pub fn slice(self: TileMap, patch: usize, data: anytype) @TypeOf(data) {
        return data[self.offsets.items[patch]..self.offsets.items[patch + 1]];
    }

    /// Returns the number of tiles in the total map.
    pub fn numTiles(self: TileMap) usize {
        return self.offsets.items[self.offsets.len - 1];
    }
};
