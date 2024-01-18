//! A module for various geometry relating data structures and algorithms, including
//! iterating over index spaces, point clustering, regions of extended blocks, etc.

// Submodules
const axis = @import("axis.zig");
const box = @import("box.zig");
const clusters = @import("clusters.zig");
const index = @import("index.zig");
const region = @import("region.zig");

// Public Exports
pub const AxisMask = axis.AxisMask;
pub const FaceIndex = axis.FaceIndex;
pub const FaceMask = axis.FaceMask;

pub const IndexBox = box.IndexBox;
pub const RealBox = box.RealBox;

pub const BlockClusters = clusters.BlockClusters;
pub const ClusterSpace = clusters.ClusterSpace;

pub const IndexMixin = index.IndexMixin;
pub const IndexSpace = index.IndexSpace;

pub const Region = region.Region;

test {
    _ = axis;
    _ = box;
    _ = clusters;
    _ = index;
    _ = region;
}
