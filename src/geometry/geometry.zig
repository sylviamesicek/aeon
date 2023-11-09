//! A module for various geometry relating data structures and algorithms, including
//! iterating over index spaces, point clustering, regions of extended blocks, etc.

// Submodules
const box = @import("box.zig");
const faces = @import("faces.zig");
const index = @import("index.zig");
const partitions = @import("partitions.zig");
const region = @import("region.zig");

// Public Exports
pub const Box = box.Box;
pub const SplitIndex = box.SplitIndex;
pub const Face = faces.Face;
pub const Index = index.Index;
pub const IndexSpace = index.IndexSpace;
pub const Partitions = partitions.Partitions;
pub const Region = region.Region;

test {
    _ = box;
    _ = faces;
    _ = index;
    _ = partitions;
    _ = region;
}
