const box = @import("box.zig");
const faces = @import("faces.zig");
const index = @import("index.zig");
const tiles = @import("tiles.zig");

pub const Box = box.Box;
pub const SplitIndex = box.SplitIndex;

pub const FaceIndex = faces.FaceIndex;

pub const IndexSpace = index.IndexSpace;

pub const Partition = tiles.Partition;
pub const Partitions = tiles.Partitions;
pub const Tiles = tiles.Tiles;

test {
    _ = box;
    _ = faces;
    _ = index;
    _ = tiles;
}
