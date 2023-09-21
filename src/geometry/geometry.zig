// Submodules
const box = @import("box.zig");
const faces = @import("faces.zig");
const space = @import("space.zig");
const partitions = @import("partitions.zig");
const region = @import("region.zig");

// Public Exports
pub const Box = box.Box;
pub const SplitIndex = box.SplitIndex;
pub const Face = faces.Face;
pub const IndexSpace = space.IndexSpace;
pub const PartitionSpace = partitions.PartitionSpace;
pub const Region = region.Region;

test {
    _ = box;
    _ = faces;
    _ = space;
    _ = partitions;
    _ = region;
}
