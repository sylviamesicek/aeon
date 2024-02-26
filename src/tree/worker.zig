const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const ArrayListUmanaged = std.ArrayListUnmanaged;
const MultiArrayList = std.MultiArrayList;
const assert = std.debug.assert;
const panic = std.debug.panic;

const manager_ = @import("manager.zig");
const tree = @import("tree.zig");
const null_index = tree.null_index;
const boundary_index = tree.boundary_index;

const common = @import("../common/common.zig");
const geometry = @import("../geometry/geometry.zig");
const utils = @import("../utils.zig");

const Range = utils.Range;
const RangeMap = utils.RangeMap;

/// Represents a generic cpu-driver node worker. This is a convience class for implementing a wide variety of routines
/// dealing with nodes (restriction, prolongation, filling ghost nodes, etc.).
pub fn NodeWorker(comptime N: usize, comptime M: usize) type {
    return struct {
        mesh: *const TreeMesh,
        manager: *const NodeManager,

        const Block = NodeManager.Block;
        const Cell = NodeManager.Cell;

        const NodeManager = manager_.NodeManager(N, M);
        const TreeMesh = tree.TreeMesh(N);

        const AxisMask = geometry.AxisMask(N);
        const FaceIndex = geometry.FaceIndex(N);
        const IndexMixin = geometry.IndexMixin(N);
        const IndexSpace = geometry.IndexSpace(N);
        const Region = geometry.Region(N);
        const addSigned = IndexMixin.addSigned;
        const coarsened = IndexMixin.coarsened;
        const mul = IndexMixin.mul;
        const refined = IndexMixin.refined;
        const scaled = IndexMixin.scaled;
        const toSigned = IndexMixin.toSigned;
        const toUnsigned = IndexMixin.toUnsigned;

        const NodeSpace = common.NodeSpace(N, M);

        const Self = @This();

        pub fn init(mesh: *const TreeMesh, manager: *const NodeManager) !@This() {
            return .{
                .mesh = mesh,
                .manager = manager,
            };
        }

        pub fn deinit(self: *@This()) void {
            _ = self;
        }

        /// Computes the l2 norm of the the field, including nodes on non-leaf cells.
        pub fn norm(self: @This(), field: []const f64) f64 {
            const manager = self.manager;

            assert(field.len == manager.numNodes());

            var result: f64 = 0.0;

            for (0..manager.numBlocks()) |block_id| {
                const block = manager.blockFromId(block_id);
                const block_field = manager.blockNodes(block_id, field);
                const node_space = self.nodeSpaceFromBlock(block);

                var cells = node_space.cellSpace().cartesianIndices();

                while (cells.next()) |cell| {
                    const v = node_space.value(cell, block_field);
                    result += v * v;
                }
            }

            return @sqrt(result);
        }

        pub fn normScaled(self: @This(), field: []const f64) f64 {
            const dimension: f64 = @floatFromInt(field.len);
            return self.norm(field) / @sqrt(dimension);
        }

        // **********************************
        // Basic Operations *****************
        // **********************************

        /// Copies src to dest.
        pub fn copy(self: @This(), dest: []f64, src: []const f64) void {
            assert(src.len == self.manager.numNodes());
            assert(dest.len == self.manager.numNodes());

            @memcpy(dest, src);
        }

        /// Copies src to dest on a particular level.
        pub fn copyLevel(self: @This(), level: usize, dest: []f64, src: []const f64) void {
            const manager = self.manager;

            assert(src.len == manager.numNodes());
            assert(dest.len == manager.numNodes());

            const blocks = manager.level_to_blocks.range(level);

            for (blocks.start..blocks.end) |block_id| {
                const block_src = manager.blockNodes(block_id, src);
                const block_dest = manager.blockNodes(block_id, dest);

                @memcpy(block_dest, block_src);
            }
        }

        /// Adds v to dest on a particular level.
        pub fn addAssignLevel(self: @This(), level: usize, dest: []f64, v: []const f64) void {
            const manager = self.manager;

            assert(v.len == manager.numNodes());
            assert(dest.len == manager.numNodes());

            const blocks = manager.level_to_blocks.range(level);

            for (blocks.start..blocks.end) |block_id| {
                const block_v = manager.blockNodes(block_id, v);
                const block_dest = manager.blockNodes(block_id, dest);

                for (0..block_dest.len) |i| {
                    block_dest[i] += block_v[i];
                }
            }
        }

        /// Subtracts v from dest on a particular level.
        pub fn subAssignLevel(self: @This(), level: usize, dest: []f64, v: []const f64) void {
            const manager = self.manager;

            assert(v.len == manager.numNodes());
            assert(dest.len == manager.numNodes());

            const blocks = manager.level_to_blocks.range(level);

            for (blocks.start..blocks.end) |block_id| {
                const block_v = manager.blockNodes(block_id, v);
                const block_dest = manager.blockNodes(block_id, dest);

                for (0..block_dest.len) |i| {
                    block_dest[i] -= block_v[i];
                }
            }
        }

        // *********************************
        // Base operations *****************
        // *********************************

        pub fn numBaseNodes(self: @This()) usize {
            return IndexSpace.fromSize(self.manager.cell_size).total();
        }

        pub fn packBase(self: @This(), dest: []f64, src: []const f64) void {
            assert(dest.len == self.numBaseNodes());
            assert(src.len == self.manager.numNodes());

            const block = self.manager.blockFromId(0);
            const block_src = self.manager.blockNodes(0, src);
            const node_space = self.nodeSpaceFromBlock(block);

            var cells = node_space.cellSpace().cartesianIndices();
            var linear: usize = 0;

            while (cells.next()) |cell| : (linear += 1) {
                const v = node_space.value(cell, block_src);
                dest[linear] = v;
            }
        }

        pub fn unpackBase(self: @This(), dest: []f64, src: []const f64) void {
            assert(src.len == self.numBaseNodes());
            assert(dest.len == self.manager.numNodes());

            const block = self.manager.blockFromId(0);
            const block_dest = self.manager.blockNodes(0, dest);
            const node_space = self.nodeSpaceFromBlock(block);

            var cells = node_space.cellSpace().cartesianIndices();
            var linear: usize = 0;

            while (cells.next()) |cell| : (linear += 1) {
                node_space.setValue(cell, block_dest, src[linear]);
            }
        }

        // **************************************
        // Refinement Helpers *******************
        // **************************************

        /// Computse refinement flags based on local truncation error. If the absolute value of the error for any node in a cell
        /// is greater than the given tolerance, that cell will be tagged for refinement.
        pub fn refineByLocalTruncErr(self: @This(), err: []const f64, tol: f64, flags: []bool) void {
            assert(flags.len == self.mesh.numCells());
            assert(err.len == self.manager.numNodes());
            assert(tol > 0);

            for (0..self.mesh.numCells()) |cell_id| {
                const block_id = self.manager.cells.items(.block)[cell_id];
                const index = self.manager.cells.items(.index)[cell_id];

                const block = self.manager.blockFromId(block_id);
                const block_err = self.manager.blockNodes(block_id, err);
                const node_space = self.nodeSpaceFromBlock(block);

                const origin = IndexMixin.mul(index, self.manager.cell_size);

                var max_err: f64 = 0.0;

                var offsets = IndexSpace.fromSize(self.manager.cell_size).cartesianIndices();

                while (offsets.next()) |offset| {
                    const cell = IndexMixin.add(origin, offset);

                    const v = node_space.value(cell, block_err);
                    max_err = @max(max_err, @abs(v));
                }

                if (max_err > tol) {
                    flags[cell_id] = true;
                }
            }
        }

        // **************************************
        // General Order Dependent Operations ***
        // **************************************

        /// Returns a namespace for all routines dependent on the order of accuracy.
        pub fn order(self: @This(), comptime O: usize) Order(O) {
            return .{
                .worker = self,
            };
        }

        pub fn Order(comptime O: usize) type {
            return struct {
                worker: Self,

                const BoundaryEngine = common.BoundaryEngine(N, M, O);
                const Engine = common.Engine(N, M, O);
                const isOperator = common.isOperator(N, M, O);
                const isProjection = common.isProjection(N, M, O);

                // **************************
                // Ghost Nodes **************
                // **************************

                /// Fills all ghost nodes.
                pub fn fillGhostNodes(self: @This(), boundary: anytype, field: []f64) void {
                    const manager = self.worker.manager;

                    assert(field.len == manager.numNodes());

                    self.fillBaseGhostNodes(boundary, field);

                    for (1..manager.numBlocks()) |block_id| {
                        self.fillBlockGhostNodes(block_id, boundary, field);
                    }
                }

                /// Fills the ghost nodes of given level.
                pub fn fillLevelGhostNodes(self: @This(), level: usize, boundary: anytype, field: []f64) void {
                    const manager = self.worker.manager;

                    assert(field.len == manager.numNodes());

                    // Short circuit in the case of level = 0
                    if (level == 0) {
                        self.fillBaseGhostNodes(boundary, field);
                    } else {
                        // Otherwise check each block on the level
                        const blocks = manager.level_to_blocks.range(level);
                        for (blocks.start..blocks.end) |block_id| {
                            self.fillBlockGhostNodes(block_id, boundary, field);
                        }
                    }
                }

                fn fillBaseGhostNodes(self: @This(), boundary: anytype, field: []f64) void {
                    const manager = self.worker.manager;

                    const block = manager.blockFromId(0);
                    const block_field = manager.blockNodes(0, field);
                    const node_space = self.worker.nodeSpaceFromBlock(block);
                    const boundary_engine = BoundaryEngine{ .space = node_space };

                    boundary_engine.fill(boundary, block_field);
                }

                /// Fills the boundary of a given block, whether by exptrapolation or prolongation.
                fn fillBlockGhostNodes(self: @This(), block_id: usize, boundary: anytype, field: []f64) void {
                    const manager = self.worker.manager;

                    assert(block_id != 0);

                    const block = manager.blockFromId(block_id);
                    const block_field = manager.blockNodes(block_id, field);
                    const node_space = self.worker.nodeSpaceFromBlock(block);
                    const boundary_engine = BoundaryEngine{ .space = node_space };

                    const regions = comptime Region.enumerateOrdered();

                    // Loop over the regions
                    inline for (comptime regions[1..]) |region| {
                        const smask = comptime region.toMask();

                        var bmask = AxisMask.initEmpty();

                        for (0..N) |axis| {
                            if (region.sides[axis] == .middle) {
                                continue;
                            }

                            const face: FaceIndex = .{
                                .axis = axis,
                                .side = region.sides[axis] == .right,
                            };

                            bmask.setValue(axis, block.boundary[face.toLinear()]);
                        }

                        // If no axes lie on a physical boundary, prolong
                        if (bmask.isEmpty()) {
                            // Prolong boundary
                            self.fillBlockGhostNodesInterior(region, block_id, field);
                        }

                        // Otherwise enumerate the possible axis combinations
                        inline for (comptime AxisMask.enumerate()) |mask| {
                            // Intersect with mask to make sure we aren't generating unnessary code.
                            const imask = comptime mask.intersectWith(smask);

                            if (comptime imask.isEmpty()) {
                                continue;
                            }

                            if (bmask.toLinear() == imask.toLinear()) {
                                boundary_engine.fillRegion(region, imask, boundary, block_field);
                            }
                        }
                    }
                }

                /// Fills ghost nodes on the interior of the numerical domain.
                fn fillBlockGhostNodesInterior(self: @This(), region: Region, block_id: usize, field: []f64) void {
                    const manager = self.worker.manager;
                    const mesh = self.worker.mesh;

                    assert(block_id != 0);

                    const block = manager.blockFromId(block_id);
                    const block_node_space = self.worker.nodeSpaceFromBlock(block);
                    const block_field: []f64 = manager.blockNodes(block_id, field);

                    // Cache pointers
                    const cells = mesh.cells.slice();
                    const cell_meta = manager.cells.slice();
                    const cell_size = manager.cell_size;

                    var cell_indices = region.innerFaceCells(block.size);

                    while (cell_indices.next()) |index| {
                        // Uses permutation to find actual cell
                        const cell = manager.cellFromBlock(block_id, index);
                        // Origin in node space
                        const origin = toSigned(mul(index, cell_size));
                        // Get neighbor in this direction
                        var neighbor = cell_meta.items(.neighbors)[cell][region.toLinear()];

                        assert(neighbor != boundary_index);

                        if (neighbor == null_index) {
                            // Prolong from coarser level.
                            const parent = cells.items(.parent)[cell];
                            // Get split mask.
                            const split = AxisMask.fromLinear(cell - cells.items(.children)[parent]);

                            const neighbor_region = region.maskedBySplit(split);

                            neighbor = cell_meta.items(.neighbors)[parent][neighbor_region.toLinear()];

                            const neighbor_block_id = cell_meta.items(.block)[neighbor];
                            const neighbor_block = manager.blockFromId(neighbor_block_id);
                            const neighbor_index = cell_meta.items(.index)[neighbor];
                            const neighbor_field: []const f64 = manager.blockNodes(neighbor_block_id, field);
                            const neighbor_node_space = self.worker.nodeSpaceFromBlock(neighbor_block);

                            var neighbor_origin = toSigned(refined(mul(neighbor_index, cell_size)));

                            for (0..N) |axis| {
                                if (split.isSet(axis)) {
                                    neighbor_origin[axis] += @intCast(cell_size[axis]);
                                }

                                switch (neighbor_region.sides[axis]) {
                                    .left => neighbor_origin[axis] += @intCast(2 * cell_size[axis]),
                                    .right => neighbor_origin[axis] -= @intCast(2 * cell_size[axis]),
                                    else => {},
                                }
                            }

                            // Iterate over node offsets
                            var offsets = region.nodes(O, cell_size);

                            while (offsets.next()) |offset| {
                                const node: [N]isize = addSigned(origin, offset);
                                const neighbor_node: [N]isize = addSigned(neighbor_origin, offset);

                                const v = neighbor_node_space.order(O).prolongCell(toUnsigned(neighbor_node), neighbor_field);
                                block_node_space.setValue(node, block_field, v);
                            }
                        } else {
                            // Neighbor is on same level.
                            const neighbor_block_id = cell_meta.items(.block)[neighbor];
                            const neighbor_index = cell_meta.items(.index)[neighbor];
                            const neighbor_block = manager.blockFromId(neighbor_block_id);
                            const neighbor_field = manager.blockNodes(neighbor_block_id, field);
                            const neighbor_node_space = self.worker.nodeSpaceFromBlock(neighbor_block);

                            var neighbor_origin = toSigned(mul(neighbor_index, cell_size));

                            for (0..N) |axis| {
                                switch (region.sides[axis]) {
                                    .left => neighbor_origin[axis] += @intCast(cell_size[axis]),
                                    .right => neighbor_origin[axis] -= @intCast(cell_size[axis]),
                                    else => {},
                                }
                            }

                            var offsets = region.nodes(O, cell_size);

                            while (offsets.next()) |offset| {
                                const node = addSigned(origin, offset);
                                const neighbor_node = addSigned(neighbor_origin, offset);

                                const v = neighbor_node_space.value(neighbor_node, neighbor_field);
                                block_node_space.setValue(node, block_field, v);
                            }
                        }
                    }
                }

                // *******************************
                // Prolongation/Restriction ******
                // *******************************

                /// Prolongs data from `level - 1` to `level`.
                pub fn prolongLevel(self: @This(), level: usize, field: []f64) void {
                    const manager = self.worker.manager;

                    assert(field.len == manager.numNodes());

                    if (level == 0) {
                        return;
                    }

                    const blocks = manager.level_to_blocks.range(level);
                    for (blocks.start..blocks.end) |block_id| {
                        self.prolongBlock(block_id, field);
                    }
                }

                fn prolongBlock(self: @This(), block_id: usize, field: []f64) void {
                    const manager = self.worker.manager;
                    const mesh = self.worker.mesh;

                    assert(block_id != 0);

                    const block = manager.blockFromId(block_id);
                    const block_field = manager.blockNodes(block_id, field);
                    const block_node_space = self.worker.nodeSpaceFromBlock(block);

                    // Cache pointers
                    const cells = mesh.cells.slice();
                    const cell_meta = manager.cells.slice();
                    const cell_size = manager.cell_size;

                    var cell_indices = IndexSpace.fromSize(block.size).cartesianIndices();

                    while (cell_indices.next()) |index| {
                        // Uses permutation to find actual cell
                        const cell = manager.cellFromBlock(block_id, index);
                        // Get parent of current cell
                        const parent = cells.items(.parent)[cell];
                        // Get split mask.
                        const split_lin = cell - cells.items(.children)[parent];
                        const split = AxisMask.fromLinear(split_lin);
                        // Get super block
                        const parent_block_id: usize = cell_meta.items(.block)[parent];
                        const parent_index: [N]usize = cell_meta.items(.index)[parent];
                        const parent_block = manager.blockFromId(parent_block_id);
                        const parent_block_field: []const f64 = manager.blockNodes(parent_block_id, field);
                        const parent_block_node_space = self.worker.nodeSpaceFromBlock(parent_block);
                        var parent_origin = refined(mul(parent_index, cell_size));
                        // Offset origin to take into account split
                        for (0..N) |axis| {
                            if (split.isSet(axis)) {
                                parent_origin[axis] += cell_size[axis];
                            }
                        }
                        // Get origin
                        const origin = mul(index, cell_size);

                        var offsets = IndexSpace.fromSize(cell_size).cartesianIndices();
                        while (offsets.next()) |offset| {
                            const node = IndexMixin.add(origin, offset);
                            const subnode = IndexMixin.add(parent_origin, offset);

                            const val = parent_block_node_space.order(O).prolongCell(subnode, parent_block_field);
                            block_node_space.setValue(node, block_field, val);
                        }
                    }
                }

                /// Restricts data from `level` to `level - 1`
                pub fn restrictLevel(self: @This(), level: usize, field: []f64) void {
                    assert(field.len == self.worker.manager.numNodes());

                    if (level == 0) {
                        return;
                    }

                    const blocks = self.worker.manager.level_to_blocks.range(level);
                    for (blocks.start..blocks.end) |block_id| {
                        self.restrictBlock(block_id, field);
                    }
                }

                fn restrictBlock(self: @This(), block_id: usize, field: []f64) void {
                    const manager = self.worker.manager;
                    const mesh = self.worker.mesh;

                    assert(block_id != 0);

                    const block = manager.blockFromId(block_id);
                    const block_field: []const f64 = manager.blockNodes(block_id, field);
                    const block_node_space = self.worker.nodeSpaceFromBlock(block);

                    // Cache pointers
                    const cells = mesh.cells.slice();
                    const cell_meta = manager.cells.slice();
                    const cell_size = manager.cell_size;

                    var cell_indices = IndexSpace.fromSize(block.size).cartesianIndices();

                    while (cell_indices.next()) |index| {
                        // Uses permutation to find actual cell
                        const cell = manager.cellFromBlock(block_id, index);
                        // Get parent of current cell
                        const parent = cells.items(.parent)[cell];
                        // Get split mask.
                        const split_lin = cell - cells.items(.children)[parent];
                        const split = AxisMask.fromLinear(split_lin);
                        // Get super block
                        const parent_block_id: usize = cell_meta.items(.block)[parent];
                        const parent_index: [N]usize = cell_meta.items(.index)[parent];
                        const parent_block = manager.blockFromId(parent_block_id);
                        const parent_block_field: []f64 = manager.blockNodes(parent_block_id, field);
                        const parent_block_node_space = self.worker.nodeSpaceFromBlock(parent_block);

                        var parent_origin = mul(parent_index, cell_size);
                        // Offset origin to take into account split
                        for (0..N) |axis| {
                            if (split.isSet(axis)) {
                                parent_origin[axis] += cell_size[axis] / 2;
                            }
                        }
                        // Get origin
                        const origin = coarsened(mul(index, cell_size));

                        var offsets = IndexSpace.fromSize(coarsened(cell_size)).cartesianIndices();
                        while (offsets.next()) |offset| {
                            const node = IndexMixin.add(parent_origin, offset);
                            const supernode = IndexMixin.add(origin, offset);

                            const val = block_node_space.order(O).restrict(supernode, block_field);
                            parent_block_node_space.setValue(node, parent_block_field, val);
                        }
                    }
                }

                // ********************************
                // Projection/Application *********
                // ********************************

                /// Using the given projection to set the values of the field.
                pub fn project(self: @This(), projection: anytype, field: []f64) void {
                    for (0..self.worker.mesh.numLevels()) |level_id| {
                        self.projectLevel(level_id, projection, field);
                    }
                }

                /// Using the given projection to set the values of the field on this level.
                pub fn projectLevel(self: @This(), level: usize, projection: anytype, field: []f64) void {
                    const Proj = @TypeOf(projection);

                    if (comptime !common.isProjection(N, M, O)(Proj)) {
                        @compileError("Projection must satisfy isProjection trait");
                    }

                    const manager = self.worker.manager;

                    assert(field.len == manager.numNodes());

                    const blocks = manager.level_to_blocks.range(level);
                    for (blocks.start..blocks.end) |block_id| {
                        self.projectBlock(block_id, projection, field);
                    }
                }

                /// Set values of the field on the given block using the projection.
                fn projectBlock(self: @This(), block_id: usize, projection: anytype, field: []f64) void {
                    const manager = self.worker.manager;

                    const block_field: []f64 = manager.blockNodes(block_id, field);
                    const block_range = manager.block_to_nodes.range(block_id);

                    const node_space = self.worker.nodeSpaceFromBlock(manager.blockFromId(block_id));

                    var cells = node_space.cellSpace().cartesianIndices();

                    while (cells.next()) |cell| {
                        const engine = Engine{
                            .space = node_space,
                            .cell = cell,
                            .start = block_range.start,
                            .end = block_range.end,
                        };

                        node_space.setValue(cell, block_field, projection.project(engine));
                    }
                }

                /// Applies the operator to the source function storing the result in field.
                pub fn apply(self: @This(), field: []f64, operator: anytype, src: []const f64) void {
                    assert(field.len == self.worker.manager.numNodes());
                    assert(src.len == self.worker.manager.numNodes());

                    for (0..self.worker.manager.numBlocks()) |block_id| {
                        self.applyBlock(block_id, field, operator, src);
                    }
                }

                /// Applies the operator to the source function on this level, storing the result in field.
                pub fn applyLevel(self: @This(), level: usize, field: []f64, operator: anytype, src: []const f64) void {
                    const manager = self.worker.manager;

                    assert(field.len == manager.numNodes());
                    assert(src.len == manager.numNodes());

                    const blocks = manager.level_to_blocks.range(level);
                    for (blocks.start..blocks.end) |block_id| {
                        self.applyBlock(block_id, field, operator, src);
                    }
                }

                /// Applies the operator to the source function on this block, storing the result on field.
                fn applyBlock(self: @This(), block_id: usize, field: []f64, operator: anytype, src: []const f64) void {
                    const Oper = @TypeOf(operator);

                    if (comptime !isOperator(Oper)) {
                        @compileError("Operator must satisfy isOperator trait");
                    }

                    const manager = self.worker.manager;

                    const block_field: []f64 = manager.blockNodes(block_id, field);
                    const block_range = manager.block_to_nodes.range(block_id);

                    const node_space = self.worker.nodeSpaceFromBlock(manager.blockFromId(block_id));

                    var cells = node_space.cellSpace().cartesianIndices();

                    while (cells.next()) |cell| {
                        const engine = Engine{
                            .space = node_space,
                            .cell = cell,
                            .start = block_range.start,
                            .end = block_range.end,
                        };

                        node_space.setValue(cell, block_field, operator.apply(engine, src));
                    }
                }

                /// Computes the residual of a given operation, storing the result in field.
                pub fn residual(self: @This(), field: []f64, rhs: []const f64, operator: anytype, src: []const f64) void {
                    assert(field.len == self.worker.manager.numNodes());
                    assert(rhs.len == self.worker.manager.numNodes());
                    assert(src.len == self.worker.manager.numNodes());

                    for (0..self.worker.manager.numBlocks()) |block_id| {
                        self.residualBlock(block_id, field, rhs, operator, src);
                    }
                }

                /// Computes the residual of a given operation on this level, storing the result in field.
                pub fn residualLevel(self: @This(), level: usize, field: []f64, rhs: []const f64, operator: anytype, src: []const f64) void {
                    const Oper = @TypeOf(operator);

                    if (comptime !isOperator(Oper)) {
                        @compileError("Operator must satisfy isOperator trait");
                    }

                    const manager = self.worker.manager;

                    assert(field.len == manager.numNodes());
                    assert(src.len == manager.numNodes());
                    assert(rhs.len == manager.numNodes());

                    const blocks = manager.level_to_blocks.range(level);
                    for (blocks.start..blocks.end) |block_id| {
                        self.residualBlock(block_id, field, rhs, operator, src);
                    }
                }

                fn residualBlock(self: @This(), block_id: usize, field: []f64, rhs: []const f64, operator: anytype, src: []const f64) void {
                    const manager = self.worker.manager;

                    const block_field: []f64 = manager.blockNodes(block_id, field);
                    const block_rhs: []const f64 = manager.blockNodes(block_id, rhs);
                    const block_range = manager.block_to_nodes.range(block_id);

                    const node_space = self.worker.nodeSpaceFromBlock(manager.blockFromId(block_id));

                    var cells = node_space.cellSpace().cartesianIndices();

                    while (cells.next()) |cell| {
                        const engine = Engine{
                            .space = node_space,
                            .cell = cell,
                            .start = block_range.start,
                            .end = block_range.end,
                        };

                        const rval = node_space.value(cell, block_rhs);

                        node_space.setValue(cell, block_field, rval - operator.apply(engine, src));
                    }
                }

                /// Computes the tau correction for this level, given a restricted residual, and a source function.
                /// Only fills field for non-leaf cells.
                pub fn tauCorrectLevel(self: @This(), level: usize, field: []f64, res: []const f64, operator: anytype, src: []const f64) void {
                    const Oper = @TypeOf(operator);

                    if (comptime !isOperator(Oper)) {
                        @compileError("Operator must satisfy isOperator trait");
                    }

                    const manager = self.worker.manager;

                    assert(field.len == manager.numNodes());
                    assert(src.len == manager.numNodes());
                    assert(res.len == manager.numNodes());

                    const blocks = manager.level_to_blocks.range(level);
                    for (blocks.start..blocks.end) |block_id| {
                        self.tauCorrectBlock(block_id, field, res, operator, src);
                    }
                }

                fn tauCorrectBlock(self: @This(), block_id: usize, field: []f64, res: []const f64, operator: anytype, src: []const f64) void {
                    const manager = self.worker.manager;

                    const block_field: []f64 = manager.blockNodes(block_id, field);
                    const block_res: []const f64 = manager.blockNodes(block_id, res);
                    const block_range = manager.block_to_nodes.range(block_id);
                    const block = manager.blockFromId(block_id);

                    const node_space = self.worker.nodeSpaceFromBlock(block);

                    const cells = self.worker.mesh.cells.slice();

                    var indices = IndexSpace.fromSize(block.size).cartesianIndices();

                    while (indices.next()) |index| {
                        const cell = manager.cellFromBlock(block_id, index);

                        if (cells.items(.children)[cell] == null_index) {
                            continue;
                        }

                        const origin = IndexMixin.mul(index, manager.cell_size);

                        var offsets = IndexSpace.fromSize(manager.cell_size).cartesianIndices();

                        while (offsets.next()) |offset| {
                            const node = IndexMixin.add(origin, offset);

                            const engine = Engine{
                                .space = node_space,
                                .cell = node,
                                .start = block_range.start,
                                .end = block_range.end,
                            };

                            const aval = operator.apply(engine, src);
                            const rval = node_space.value(node, block_res);

                            node_space.setValue(node, block_field, rval + aval);
                        }
                    }
                }

                // ***************************************
                // Smoothing *****************************
                // ***************************************

                /// Performs jacobi smoothing on this level.
                pub fn smoothLevel(self: @This(), level: usize, field: []f64, operator: anytype, src: []const f64, rhs: []const f64) void {
                    const Oper = @TypeOf(operator);

                    if (comptime !common.isOperator(N, M, O)(Oper)) {
                        @compileError("Operator must satisfy isOperator trait");
                    }

                    const manager = self.worker.manager;

                    assert(field.len == manager.numNodes());
                    assert(src.len == manager.numNodes());

                    const blocks = manager.level_to_blocks.range(level);
                    for (blocks.start..blocks.end) |block_id| {
                        self.smoothBlock(block_id, field, operator, src, rhs);
                    }
                }

                /// Performs jacobi smoothing on this block.
                fn smoothBlock(self: @This(), block_id: usize, field: []f64, operator: anytype, src: []const f64, rhs: []const f64) void {
                    const manager = self.worker.manager;

                    const block_field: []f64 = manager.blockNodes(block_id, field);
                    const block_rhs: []const f64 = manager.blockNodes(block_id, rhs);
                    const block_src: []const f64 = manager.blockNodes(block_id, src);
                    const block_range = manager.block_to_nodes.range(block_id);

                    const node_space = self.worker.nodeSpaceFromBlock(manager.blockFromId(block_id));

                    var cells = node_space.cellSpace().cartesianIndices();

                    while (cells.next()) |cell| {
                        const engine = Engine{
                            .space = node_space,
                            .cell = cell,
                            .start = block_range.start,
                            .end = block_range.end,
                        };

                        const sval = node_space.value(cell, block_src);
                        const rval = node_space.value(cell, block_rhs);
                        const app = operator.apply(engine, src);
                        const appDiag = operator.applyDiag(engine);

                        node_space.setValue(cell, block_field, sval + 2.0 / 3.0 * (rval - app) / appDiag);
                    }
                }

                // **********************************
                // Dissipation

                /// Applies dissipation to a first order hyperbolic system of equations.
                /// Here stencils have been scaled such that, in order to perserve stability,
                /// `eps` must be less than or equal to `min(delta_x)/(delta_t)`, which may be written
                /// `eps <= 1/max(CFL)`.
                pub fn dissipation(self: @This(), eps: f64, deriv: []f64, src: []const f64) void {
                    const manager = self.worker.manager;

                    assert(deriv.len == manager.numNodes());
                    assert(src.len == manager.numNodes());

                    for (0..manager.numBlocks()) |block_id| {
                        self.dissipationBlock(block_id, eps, deriv, src);
                    }
                }

                pub fn dissipationLevel(self: @This(), level: usize, eps: f64, deriv: []f64, src: []const f64) void {
                    const manager = self.worker.manager;

                    assert(deriv.len == manager.numNodes());
                    assert(src.len == manager.numNodes());

                    const blocks = manager.level_to_blocks.range(level);
                    for (blocks.start..blocks.end) |block_id| {
                        self.dissipationBlock(block_id, eps, deriv, src);
                    }
                }

                fn dissipationBlock(self: @This(), block_id: usize, eps: f64, deriv: []f64, src: []const f64) void {
                    const manager = self.worker.manager;

                    const block_deriv: []f64 = manager.blockNodes(block_id, deriv);
                    const block_src: []const f64 = manager.blockNodes(block_id, src);
                    const block = manager.blockFromId(block_id);

                    var spacing: [N]f64 = undefined;

                    for (0..N) |axis| {
                        spacing[axis] = block.bounds.size[axis];
                        spacing[axis] /= @floatFromInt(block.size[axis] * manager.cell_size[axis]);
                    }

                    const node_space = self.worker.nodeSpaceFromBlock(block);

                    var cells = node_space.cellSpace().cartesianIndices();

                    while (cells.next()) |cell| {
                        const v = node_space.value(cell, block_deriv);

                        var result: f64 = 0.0;

                        inline for (0..N) |axis| {
                            result += eps / spacing[axis] * node_space.order(O).dissipation(axis, cell, block_src);
                        }

                        node_space.setValue(cell, block_deriv, v + result);
                    }
                }
            };
        }

        /// Helper function for determining the node space of a block
        fn nodeSpaceFromBlock(self: @This(), block: Block) NodeSpace {
            var size: [N]usize = undefined;

            for (0..N) |axis| {
                size[axis] = block.size[axis] * self.manager.cell_size[axis];
            }

            return .{
                .bounds = block.bounds,
                .size = size,
            };
        }
    };
}
