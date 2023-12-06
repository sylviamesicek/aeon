const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const ArrayListUnmanaged = std.ArrayListUnmanaged;
const MultiArrayList = std.MultiArrayList;
const ArenaAllocator = std.heap.ArenaAllocator;
const assert = std.debug.assert;
const exp2 = std.math.exp2;
const maxInt = std.math.maxInt;

// Root imports

const basis = @import("../basis/basis.zig");
const geometry = @import("../geometry/geometry.zig");
const mesh = @import("../mesh/mesh.zig");
const nodes = @import("../nodes/nodes.zig");

const CellMap = mesh.CellMap;
const TileMap = mesh.TileMap;

const NodeMap = nodes.NodeMap;
const NodeSpace = nodes.NodeSpace;

// ************************
// Mesh *******************
// ************************

/// A contigious and uniformly rectangular block of data on the mesh.
pub fn Block(comptime N: usize) type {
    return struct {
        /// Bounds of this block
        bounds: IndexBox,
        /// Patch this block belongs to
        patch: usize,

        const IndexBox = geometry.IndexBox(N);
    };
}

/// A grouping of multiple nearby blocks on a single level, arranged into a hierarchy between levels.
pub fn Patch(comptime N: usize) type {
    return struct {
        bounds: IndexBox,
        level: usize,
        parent: ?usize,
        children_offset: usize,
        children_total: usize,
        block_offset: usize,
        block_total: usize,

        const IndexBox = geometry.IndexBox(N);
    };
}

/// A collection of blocks and patches with the same grid spacing.
pub fn Level(comptime N: usize) type {
    return struct {
        tile_size: [N]usize,
        patch_offset: usize,
        patch_total: usize,
        block_offset: usize,
        block_total: usize,
    };
}

/// Represents a block structured mesh, ie a numerical discretisation of a physical domain into a number of levels,
/// each of which consists of patches of tiles, and blocks of cells. This class handles accessing and
/// storing data for each cell/tile, allocating ghost cells, and running regridding and
/// transfer algorithms.
pub fn Mesh(comptime N: usize) type {
    return struct {
        /// Allocator used for various arraylists stored in this struct.
        gpa: Allocator,
        /// THe physical box that this mesh covers.
        physical_bounds: RealBox,
        /// The dimensions of the mesh in index space.
        index_size: [N]usize,
        /// The number of cells along each time edge.
        tile_width: usize,

        // Block ArrayListUnmanaged
        block_capacity: usize,
        blocks: []Block(N),
        // Patch ArrayListUnmanaged
        patch_capacity: usize,
        patches: []Patch(N),
        // Level ArrayListUnmanaged
        level_capacity: usize,
        levels: []Level(N),

        /// Configuration for a mesh.
        pub const Config = struct {
            physical_bounds: RealBox,
            index_size: [N]usize,
            tile_width: usize,

            /// Checks that the given mesh config is valid.
            pub fn check(self: Config) void {
                assert(self.tile_width >= 1);
                for (0..N) |i| {
                    assert(self.index_size[i] > 0);
                    assert(self.physical_bounds.size[i] > 0.0);
                }
            }
        };

        // Aliases
        const Self = @This();
        const IndexBox = geometry.IndexBox(N);
        const RealBox = geometry.RealBox(N);
        const FaceIndex = geometry.FaceIndex(N);
        const IndexSpace = geometry.IndexSpace(N);
        const Region = geometry.Region(N);

        // Mixins
        const IndexMixin = geometry.IndexMixin(N);

        const add = IndexMixin.add;
        const sub = IndexMixin.sub;
        const scaled = IndexMixin.scaled;
        const splat = IndexMixin.splat;
        const toSigned = IndexMixin.toSigned;

        /// Initialises a new mesh with an general purpose allocator, subject to the given configuration.
        pub fn init(allocator: Allocator, config: Config) !Self {
            // Check config
            config.check();

            var blocks: ArrayListUnmanaged(Block(N)) = .{};
            errdefer blocks.deinit(allocator);

            var patches: ArrayListUnmanaged(Patch(N)) = .{};
            errdefer patches.deinit(allocator);

            var levels: ArrayListUnmanaged(Level(N)) = .{};
            errdefer levels.deinit(allocator);

            try blocks.append(allocator, .{
                .bounds = .{
                    .origin = splat(0),
                    .size = config.index_size,
                },
                .patch = 0,
            });

            try patches.append(allocator, .{
                .bounds = .{
                    .origin = splat(0),
                    .size = config.index_size,
                },
                .level = 0,
                .parent = null,
                .children_offset = 0,
                .children_total = 0,
                .block_offset = 0,
                .block_total = 1,
            });

            try levels.append(allocator, .{
                .tile_size = config.index_size,
                .patch_offset = 0,
                .patch_total = 1,
                .block_offset = 0,
                .block_total = 1,
            });

            return .{
                .gpa = allocator,
                .index_size = config.index_size,
                .physical_bounds = config.physical_bounds,
                .tile_width = config.tile_width,
                .block_capacity = blocks.capacity,
                .patch_capacity = patches.capacity,
                .level_capacity = levels.capacity,
                .blocks = blocks.items,
                .patches = patches.items,
                .levels = levels.items,
            };
        }

        /// Deinitalises a mesh.
        pub fn deinit(self: *Self) void {
            var blocks: ArrayListUnmanaged(Block(N)) = .{
                .capacity = self.block_capacity,
                .items = self.blocks,
            };

            var patches: ArrayListUnmanaged(Patch(N)) = .{
                .capacity = self.patch_capacity,
                .items = self.patches,
            };

            var levels: ArrayListUnmanaged(Level(N)) = .{
                .capacity = self.level_capacity,
                .items = self.levels,
            };

            levels.deinit(self.gpa);
            patches.deinit(self.gpa);
            blocks.deinit(self.gpa);
        }

        // *************************
        // Block Map ***************
        // *************************

        /// Builds a tile vector of tags from a cell vector of tags. This is used to
        /// construct tile tags from per-cell refinement criteria (for instance the local gradient
        /// difference with underlying mesh, etc.), and is simply the composite `or` operator
        /// applied to every cell for each tile.
        pub fn buildTagsFromCells(self: *const Self, dest: []bool, src: []const bool) void {
            @memset(dest, false);

            for (self.blocks) |block| {
                const patch = self.patches[block.patch];

                const block_src = src[block.cell_offset .. block.cell_offset + block.cell_total];
                const block_dest = dest[patch.tile_offset .. patch.tile_offset + patch.tile_total];

                const patch_space = IndexSpace.fromBox(patch.bounds);
                const block_space = IndexSpace.fromBox(block.bounds);
                const cell_space = IndexSpace.fromSize(scaled(block.bounds.size, self.tile_width));

                var tiles = block_space.cartesianIndices();

                while (tiles.next()) |tile| {
                    const origin = scaled(tile, self.tile_width);

                    const patch_tile = patch.bounds.localFromGlobal(block.bounds.globalFromLocal(tile));
                    const patch_linear = patch_space.linearFromCartesian(patch_tile);

                    var cells = IndexSpace.fromSize(splat(self.tile_width)).cartesianIndices();

                    while (cells.next()) |cell| {
                        const global_cell = add(origin, cell);

                        if (block_src[cell_space.linearFromCartesian(global_cell)]) {
                            block_dest[patch_linear] = true;
                            break;
                        }
                    }
                }
            }
        }

        // /// Builds a block child map. ie a map from each tile in the mesh to the id of the block that covers it
        // /// on the next level, if any.
        // pub fn buildBlockChildMap(self: *const Self, map: []usize) void {
        //     assert(map.len == self.tile_total);

        //     @memset(map, maxInt(usize));

        //     for (0..self.active_levels - 1) |l| {
        //         const level = self.getLevel(l);
        //         const refined = self.getLevel(l + 1);

        //         for (0..level.patchTotal()) |patch| {
        //             const patch_bounds: IndexBox = level.patches.items(.bounds)[patch];
        //             const patch_map: []usize = self.patchTileSlice(l, patch, map);
        //             for (level.childrenSlice(patch)) |child_patch| {
        //                 const offset = refined.patches.items(.block_offset)[child_patch];
        //                 const total = refined.patches.items(.block_total)[child_patch];
        //                 for (offset..offset + total) |child_block| {
        //                     var bounds = refined.blocks.items(.bounds)[child_block];
        //                     bounds.coarsen();

        //                     IndexSpace.fromBox(patch_bounds).fillWindow(bounds.relativeTo(patch), usize, patch_map, child_block);
        //                 }
        //             }
        //         }

        //         for (level.blocks.items(.bounds), level.blocks.items(.patch), 0..) |bounds, patch, id| {
        //             const pbounds: IndexBox = level.patches.items(.bounds)[patch];
        //             const tile_to_block: []usize = self.patchTileSlice(l, patch, map);
        //             IndexSpace.fromBox(pbounds).fillWindow(bounds.relativeTo(pbounds), usize, tile_to_block, id);
        //         }
        //     }
        // }

        // pub fn buildTransferMap(self: *const Self, map: []usize) !void {
        //     assert(map.len == self.transferTotal());

        //     @memset(map, std.math.maxInt(usize));
        //     @memset(self.baseTransferSlice(usize, map), 0);

        //     for (self.levels.items, 0..) |*level, l| {
        //         const level_map: []usize = self.levelTransferSlice(l, usize, map);

        //         for (level.transfer_blocks.items(.bounds), level.transfer_blocks.items(.patch), 0..) |bounds, parent, id| {
        //             const pbounds: IndexBox = level.transfer_patches.items(.bounds)[parent];
        //             const tile_offset: usize = level.transfer_patches.items(.tile_offset)[parent];
        //             const tile_total: usize = level.transfer_patches.items(.tile_total)[parent];
        //             const tile_to_block: []usize = level_map[tile_offset..(tile_offset + tile_total)];

        //             pbounds.space().fillSubspace(bounds.relativeTo(pbounds), usize, tile_to_block, id);
        //         }
        //     }
        // }

        // ***************************
        // Physical Bounds ***********
        // ***************************

        /// Computes the physical bounds of a block
        pub fn blockPhysicalBounds(self: *const Self, block: usize) RealBox {
            const patch = self.blocks[block].patch;
            const level = self.patches[patch].level;
            const index_size: [N]usize = self.levels[level].tile_size;

            // Get bounds of this block
            const bounds = self.blocks[block].bounds;

            var physical_bounds: RealBox = undefined;

            for (0..N) |i| {
                const sratio: f64 = @as(f64, @floatFromInt(bounds.size[i])) / @as(f64, @floatFromInt(index_size[i]));
                const oratio: f64 = @as(f64, @floatFromInt(bounds.origin[i])) / @as(f64, @floatFromInt(index_size[i]));

                physical_bounds.size[i] = self.physical_bounds.size[i] * sratio;
                physical_bounds.origin[i] = self.physical_bounds.origin[i] + self.physical_bounds.size[i] * oratio;
            }

            return physical_bounds;
        }

        /// Computes the number of cells along each axis for a given block.
        pub fn blockCellSize(self: *const Self, block: usize) [N]usize {
            return scaled(self.blocks[block].bounds.size, self.tile_width);
        }

        /// Finds the level a block resides on.
        pub fn blockLevel(self: *const Self, block: usize) usize {
            return self.patches[self.blocks[block].patch].level;
        }

        // *********************************
        // Maps ****************************
        // *********************************

        /// Caches node offsets for each block.
        pub fn buildNodeMap(self: *const Self, comptime M: usize, node_map: *NodeMap) !void {
            try node_map.offsets.resize(self.blocks.len + 1);

            var off: usize = 0;
            var cur: usize = 0;

            for (0..self.blocks.len) |block| {
                const total = NodeSpace(N, M).fromCellSize(self.blockCellSize(block)).numNodes();

                node_map.offsets.items[cur] = off;

                off += total;
                cur += 1;
            }

            node_map.offsets.items[cur] = off;
        }

        /// Caches cell offsets for each block.
        pub fn buildCellMap(self: *const Self, cell_map: *CellMap) !void {
            try cell_map.offsets.resize(self.blocks.len + 1);

            var off: usize = 0;
            var cur: usize = 0;

            for (0..self.blocks.len) |block| {
                const total = IndexSpace.fromSize(self.blockCellSize(block)).total();

                cell_map.offsets.items[cur] = off;

                off += total;
                cur += 1;
            }

            cell_map.offsets.items[cur] = off;
        }

        /// Caches tile offsets for each block.
        pub fn buildTileMap(self: *const Self, tile_map: *TileMap) !void {
            try tile_map.offsets.resize(self.patches.len + 1);

            var off: usize = 0;
            var cur: usize = 0;

            for (self.patches) |patch| {
                const total = IndexSpace.fromBox(patch.bounds).total();

                tile_map.offsets.items[cur] = off;

                off += total;
                cur += 1;
            }

            tile_map.offsets.items[cur] = off;
        }

        // *********************************
        // Tile to Block ********************
        // *********************************

        /// Builds a block map, ie a map from each tile in the mesh to the id of the block that tile is in.
        pub fn tileToBlock(self: *const Self, tile_map: TileMap, tile_to_blocks: []usize) void {
            @memset(self.block_map, maxInt(usize));

            for (self.patches, 0..) |patch, patch_id| {
                const tile_to_block: []usize = tile_map.slice(patch_id, tile_to_blocks);

                for (patch.block_offset..patch.block_total + patch.block_offset) |block_id| {
                    const block = self.blocks[block_id];
                    IndexSpace.fromBox(patch.bounds).fillWindow(block.bounds.relativeTo(patch.bounds), usize, tile_to_block, block_id);
                }
            }
        }
    };
}
