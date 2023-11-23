//! A module for various geometry relating data structures and algorithms, including
//! iterating over index spaces, point clustering, regions of extended blocks, etc.

// Submodules
const box = @import("box.zig");
const index = @import("index.zig");
const partitions = @import("partitions.zig");
const region = @import("region.zig");

// Public Exports
pub const FaceIndex = box.FaceIndex;
pub const IndexBox = box.IndexBox;
pub const RealBox = box.RealBox;
pub const SplitIndex = box.SplitIndex;
pub const numFaces = box.numFaces;

pub const IndexMixin = index.IndexMixin;
pub const IndexSpace = index.IndexSpace;

pub const Partitions = partitions.Partitions;
pub const Region = region.Region;

test {
    _ = box;
    _ = index;
    _ = partitions;
    _ = region;
}
