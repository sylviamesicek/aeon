const box = @import("box.zig");
const faces = @import("faces.zig");
const index = @import("index.zig");
const partitions = @import("partitions.zig");
const region = @import("region.zig");

pub const Box = box.Box;
pub const SplitIndex = box.SplitIndex;

pub const Face = faces.Face;

pub const IndexSpace = index.IndexSpace;

pub const PartitionSpace = partitions.PartitionSpace;

pub const Region = region.Region;

test {
    _ = box;
    _ = faces;
    _ = index;
    _ = partitions;
    _ = region;
}
