//! A module for various geometry relating data structures and algorithms, including
//! iterating over index spaces, point clustering, regions of extended blocks, etc.

// Submodules
const box = @import("box.zig");
const clusters = @import("clusters.zig");
const index = @import("index.zig");
const region = @import("region.zig");

// Public Exports
pub const AxisMask = box.AxisMask;
pub const FaceIndex = box.FaceIndex;
pub const FaceMask = box.FaceMask;
pub const IndexBox = box.IndexBox;
pub const RealBox = box.RealBox;
pub const SplitIndex = box.SplitIndex;

pub const BlockClusters = clusters.BlockClusters;
pub const ClusterSpace = clusters.ClusterSpace;

pub const IndexMixin = index.IndexMixin;
pub const IndexSpace = index.IndexSpace;

pub const Region = region.Region;

test {
    _ = box;
    _ = clusters;
    _ = index;
    _ = region;
}
