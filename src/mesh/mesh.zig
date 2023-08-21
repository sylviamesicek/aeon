const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const ArrayListUnmanaged = std.ArrayListUnmanaged;
const ArenaAllocator = std.heap.ArenaAllocator;
const assert = std.debug.assert;
const exp2 = std.math.exp2;

const geometry = @import("../geometry/geometry.zig");
const Box = geometry.Box;
const Geometry = geometry.Geometry;
const IndexSpace = geometry.IndexSpace;
const UniformGeometry = geometry.UniformGeometry;
const Tiles = geometry.Tiles;
const Partitions = geometry.Partitions;

pub const TileSrc = enum(u2) {
    unchanged,
    /// If this bit is set, it indicates that the index points
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
    return struct {
        gpa: Allocator,
        base: UniformGeometry(N),
        levels: ArrayListUnmanaged(Level),
        index_size: [N]usize,
        global_refinement: usize,
        total_tiles: usize,
        total_cells: usize,

        // Aliases
        const Self = @This();

        pub const Config = struct {
            physical_bounds: Box(N, f64),
            index_size: [N]usize,
            tile_width: usize,
            ghost_width: usize,
            global_refinement: usize,
        };

        pub const Level = struct {
            /// Geometry corresponding to this level.
            geometry: Geometry(N),
            /// Stores for each patch, the parent patch in level `l - 1`
            parents: ArrayListUnmanaged(usize),
            /// Stores for each patch, the children patches in level `l + 1`
            children: ArrayListUnmanaged([]usize),
            /// Index buffer (which children array points in to).
            index_buffer: ArrayListUnmanaged(usize),
            tile_offset: usize,
            cell_offset: usize,
        };

        pub fn init(allocator: Allocator, config: Config) Self {
            // Scale initial size by 2^global_refinement
            const scale: usize = exp2(config.global_refinement);

            var base_size: [N]usize = undefined;

            for (0..N) |axis| {
                base_size[axis] = scale * config.index_size[axis];
            }

            // Construct the base level
            const base = UniformGeometry(N){
                .physical_bounds = config.physical_bounds,
                .index_size = base_size,
                .tile_width = config.tile_width,
                .ghost_width = config.ghost_width,
            };

            return .{
                .gpa = allocator,
                .base = base,
                .levels = ArrayListUnmanaged(Level){},
                .index_size = config.index_size,
                .global_refinement = config.global_refinement,
                .total_tiles = base.tileTotal(),
                .total_cells = base.cellTotal(),
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.levels.items) |*level| {
                level.geometry.deinit();
                level.parents.deinit(self.gpa);
                level.children.deinit(self.gpa);
                level.index_buffer.deinit(self.gpa);
            }

            self.levels.deinit(self.gpa);
        }

        pub const RegridConfig = struct {
            max_level: usize,
            block_max_tiles: usize,
            block_efficiency: f64,
            patch_max_tiles: usize,
            patch_effciency: f64,
        };

        pub fn regrid(self: *Self, map: *ArrayList(TileMap), src: Self, tags: []const usize, config: RegridConfig) !void {
            // TODO: Ensure proper nesting by preprocessing tags.
            // TODO: Generate clusters correctly and update cluster heirarchy

            // 1. Copy base level, clear refined levels, and clear map.
            // ********************************************
            self.base = src.base;
            self.clearRetainingCapacity();
            map.clearRetainingCapacity();

            // 2. Set refined levels with appropriate settings.
            // ************************************************
            try self.ensureLength(config.max_level);
            self.propagateBaseSettings();

            // 3. Generate Levels on new mesh.
            // *******************************
            //     a. The tags that define level `l` are stored on
            //        the old level `l - 1`.
            //     b. Generate clusters by taking patches on `l+1`
            //        and scaling them to be on level `l-1`
            //     c. Run partitioning algorithm using these clusters
            //        and low target efficiency.
            //     d. Translate resulting patches to level `l` and mark
            //        tags appropriately.
            //     e. Run partitioning algorithm on these patches using
            //        higher efficiency.

            // Use arena allocator to cache memory allocations
            var arena: ArenaAllocator = ArenaAllocator.init(self.gpa);
            defer arena.deinit();

            var allocator: Allocator = arena.allocator();

            // Iterate each level in reverse order.
            for (0..config.max_level) |l| {
                // Index of the current level
                const level_index: usize = config.max_level - 1 - l;

                // Get pointers to coarse level of old mesh and refined level of new mesh.
                const coarse: *const Level = &src.levels.items[level_index - 1];
                var level: *Level = &self.levels.items[level_index];
                var refined: *Level = &self.levels.items[level_index + 1];

                // Offset into tags array.
                const level_tags = tags[coarse.tile_offset..];

                // Iterate the patches of the coarse level.
                for (coarse.geometry.patches.items(.bounds), coarse.geometry.patches.items(.offset)) |bounds, offset| {
                    // Every "frame" reset arena.
                    defer arena.reset(.retain_capacity);

                    // Space corresponding to coarse level patch.
                    const space = bounds.space();

                    // Get a mutable slice to tags corresponding to this patch
                    const coarse_tags = level_tags[offset..(offset + space.total())];

                    // Run point clustering algorithm to determine new patches
                    var partitions: Partitions(N) = .{};
                    defer partitions.deinit(allocator);

                    // Build clusters array
                    var clusters = allocator.alloc(Box(N, usize), refined.geometry.patches.len);
                    defer allocator.free(clusters);

                    @memcpy(clusters, refined.geometry.patches.items(.bounds));

                    for (clusters) |*cluster| {
                        cluster.coarsen();
                        cluster.coarsen();
                    }

                    try partitions.compute(
                        allocator,
                        .{
                            .size = bounds.widths,
                            .tags = coarse_tags,
                            .clusters = clusters,
                        },
                        config.patch_max_tiles,
                        config.patch_effciency,
                    );

                    try level.index_buffer.appendSlice(self.gpa, partitions.buffer.items);
                    try level.children.appendSlice(self.gpa, partitions.blocks.items(.children));

                    // Buffer for tags on new patches
                    var patch_tags = ArrayListUnmanaged(usize){};
                    defer patch_tags.deinit(allocator);
                    // Buffer for blocks on new patches
                    var patch_blocks = ArrayListUnmanaged(Box(N, usize)){};
                    defer patch_blocks.deinit(allocator);
                    // Cache object for building blocks on new patches.
                    var patch_partitions: Partitions(N) = .{};
                    defer patch_partitions.deinit(allocator);

                    // Iterate newly computed patches
                    for (partitions.blocks.items(.bounds)) |patch| {
                        // Represents patch space on coarse level
                        const patch_space = patch.space();
                        patch_tags.resize(allocator, patch_space.total());

                        space.window(patch, bool, patch_tags.items, coarse_tags);

                        // Perform point clustering.
                        try patch_partitions.compute(
                            allocator,
                            .{
                                .size = patch.widths,
                                .tags = patch_tags,
                                .clusters = &[_]usize{},
                            },
                            config.block_max_tiles,
                            config.block_efficiency,
                        );

                        // Translate from coarse to refined indices (scale by two).
                        patch_blocks.resize(allocator, patch_partitions.blocks.len);
                        @memcpy(patch_blocks.items, patch_partitions.blocks.items(.bounds));

                        for (patch_blocks.items) |*block| {
                            block.refine();
                        }

                        var patch_refined: Box(N, usize) = patch;
                        patch_refined.refine();

                        // Add this new patch to the refined geometry.
                        level.geometry.addPatch(patch_refined, patch_blocks.items);
                    }
                }
            }

            // 4. Recompute level offsets and totals.
            // **************************************

            var tile_offset: usize = 0;
            var cell_offset: usize = 0;

            for (self.levels.items) |*level| {
                level.tile_offset = tile_offset;
                level.cell_offset = cell_offset;

                tile_offset += level.geometry.tileTotal();
                cell_offset += level.geometry.cellTotal();
            }

            self.total_cells = cell_offset;
            self.total_tiles = tile_offset;
        }

        pub fn transfer(self: Self, map: ArrayList(TileMap), src: Self, new: *ArrayList(f64), old: ArrayList(f64)) !void {
            _ = old;
            _ = new;
            _ = src;
            _ = map;
            _ = self;
        }

        fn clearRetainingCapacity(self: *Self) void {
            for (self.levels.items) |*level| {
                level.geometry.clearRetainingCapacity();
                level.parents.clearRetainingCapacity();
                level.children.clearRetainingCapacity();
                level.index_buffer.clearRetainingCapacity();
            }
        }

        fn ensureLength(self: *Self, levels: usize) !void {
            if (levels > self.levels.items.len) {
                const n = levels - self.levels.items.len;
                const geom = Geometry(N).init(self.gpa, .{
                    .physical_bounds = .{
                        .origin = [1]f64{0.0} ** N,
                        .widths = [1]usize{0} ** N,
                    },
                    .index_size = [1]usize{0} ** N,
                    .tile_width = 0,
                    .ghost_width = 0,
                });

                try self.levels.appendNTimes(self.gpa, .{
                    .geometry = geom,
                    .parents = ArrayListUnmanaged(usize){},
                    .children = ArrayListUnmanaged(usize){},
                    .index_buffer = ArrayListUnmanaged(usize){},
                    .tile_offset = 0,
                    .cell_offset = 0,
                }, n);
            }
        }

        fn propagateBaseSettings(self: *Self) !void {
            // Now set physical bounds, index size, tile width and ghost width.
            for (self.levels.items, 1..) |*level, i| {
                level.geometry.physical_bounds = self.base.physical_bounds;
                level.geometry.index_size = self.base.index_size;
                level.geometry.tile_width = self.base.tile_width;
                level.geometry.ghost_width = self.base.ghost_width;

                const scale: usize = exp2(i);

                for (0..N) |axis| {
                    level.geometry.index_size[axis] *= scale;
                }
            }
        }
    };
}
