const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const ArrayListUmanaged = std.ArrayListUnmanaged;
const MultiArrayList = std.MultiArrayList;
const assert = std.debug.assert;
const panic = std.debug.panic;

const manager_ = @import("manager.zig");
const mesh_ = @import("mesh.zig");
const null_index = mesh_.null_index;
const boundary_index = mesh_.boundary_index;

const common = @import("../common/common.zig");
const geometry = @import("../geometry/geometry.zig");
const utils = @import("../utils.zig");

const BoundaryEngine = common.BoundaryEngine;
const Engine = common.Engine;
const checkOperator = common.checkFunction;
const checkFunction = common.checkFunction;
const checkBoundarySet = common.checkBoundarySet;

const Range = utils.Range;
const RangeMap = utils.RangeMap;

/// Represents a generic cpu-driver node worker. This is a convience class for implementing a wide variety of routines
/// dealing with nodes (restriction, prolongation, filling ghost nodes, etc.).
pub fn NodeWorker(comptime N: usize, comptime M: usize) type {
    return struct {
        mesh: *const Mesh,
        manager: *const NodeManager,

        const Block = NodeManager.Block;
        const Cell = NodeManager.Cell;

        const NodeManager = manager_.NodeManager(N, M);
        const Mesh = mesh_.Mesh(N);

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

        pub fn init(mesh: *const Mesh, manager: *const NodeManager) !@This() {
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
                const block_field = manager.blockNodes(block_id, field);
                const node_space = self.blockNodeSpace(block_id);

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

            const block_src = self.manager.blockNodes(0, src);
            const node_space = self.blockNodeSpace(0);

            var vertices = node_space.vertexSpace().cartesianIndices();
            var linear: usize = 0;

            while (vertices.next()) |vertex| : (linear += 1) {
                const v = node_space.vertexValue(vertex, block_src);
                dest[linear] = v;
            }
        }

        pub fn unpackBase(self: @This(), dest: []f64, src: []const f64) void {
            assert(src.len == self.numBaseNodes());
            assert(dest.len == self.manager.numNodes());

            const block_dest = self.manager.blockNodes(0, dest);
            const node_space = self.blockNodeSpace(0);

            var vertices = node_space.vertexSpace().cartesianIndices();
            var linear: usize = 0;

            while (vertices.next()) |vertex| : (linear += 1) {
                node_space.setVertexValue(vertex, block_dest, src[linear]);
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

                const block_err = self.manager.blockNodes(block_id, err);
                const node_space = self.blockNodeSpace(block_id);

                const origin = IndexMixin.mul(index, self.manager.cell_size);

                var max_err: f64 = 0.0;

                var offsets = IndexSpace.fromSize(self.manager.cell_size).cartesianIndices();

                while (offsets.next()) |offset| {
                    const vertex = IndexMixin.add(origin, offset);

                    const v = node_space.vertexValue(vertex, block_err);
                    max_err = @max(max_err, @abs(v));
                }

                if (max_err > tol) {
                    flags[cell_id] = true;
                }
            }
        }

        // ********************************
        // Projection/Application *********
        // ********************************

        /// Using the given projection to set the values of the field.
        pub fn project(self: @This(), projection: anytype, field: []f64) void {
            for (0..self.mesh.numLevels()) |level_id| {
                self.projectLevel(level_id, projection, field);
            }
        }

        /// Using the given projection to set the values of the field on this level.
        pub fn projectLevel(self: @This(), level: usize, projection: anytype, field: []f64) void {
            assert(field.len == self.manager.numNodes());

            const blocks = self.manager.level_to_blocks.range(level);
            for (blocks.start..blocks.end) |block_id| {
                self.projectBlock(block_id, projection, field);
            }
        }

        /// Set values of the field on the given block using the projection.
        fn projectBlock(self: @This(), block_id: usize, projection: anytype, field: []f64) void {
            const Proj: type = @TypeOf(projection);

            checkFunction(N, M, Proj);

            const block_field: []f64 = self.manager.blockNodes(block_id, field);
            const block_range = self.manager.block_to_nodes.range(block_id);

            const node_space = self.blockNodeSpace(block_id);

            var nodes = node_space.nodes(0);

            while (nodes.next()) |node| {
                const engine = Engine(N, M, Proj.order){
                    .space = node_space,
                    .node = node,
                    .range = block_range,
                };

                node_space.setValue(node, block_field, projection.eval(engine));
            }
        }

        /// Applies the operator to the source function storing the result in field.
        pub fn apply(self: @This(), field: []f64, operator: anytype, src: []const f64) void {
            assert(field.len == self.manager.numNodes());
            assert(src.len == self.manager.numNodes());

            for (0..self.manager.numBlocks()) |block_id| {
                self.applyBlock(block_id, field, operator, src);
            }
        }

        /// Applies the operator to the source function on this level, storing the result in field.
        pub fn applyLevel(self: @This(), level: usize, field: []f64, operator: anytype, src: []const f64) void {
            assert(field.len == self.manager.numNodes());
            assert(src.len == self.manager.numNodes());

            const blocks = self.manager.level_to_blocks.range(level);
            for (blocks.start..blocks.end) |block_id| {
                self.applyBlock(block_id, field, operator, src);
            }
        }

        /// Applies the operator to the source function on this block, storing the result on field.
        fn applyBlock(self: @This(), block_id: usize, field: []f64, operator: anytype, src: []const f64) void {
            const Oper: type = @TypeOf(operator);

            checkOperator(N, M, Oper);

            const block_field: []f64 = self.manager.blockNodes(block_id, field);
            const block_range = self.manager.block_to_nodes.range(block_id);

            const node_space = self.blockNodeSpace(block_id);

            var nodes = node_space.nodes(0);

            while (nodes.next()) |node| {
                const engine = Engine(N, M, Oper.order){
                    .space = node_space,
                    .node = node,
                    .range = block_range,
                };

                node_space.setValue(node, block_field, operator.apply(engine, src));
            }
        }

        // ***************************
        // Residuals *****************
        // ***************************

        /// Computes the residual of a given operation, storing the result in field.
        pub fn residual(self: @This(), field: []f64, rhs: []const f64, operator: anytype, src: []const f64) void {
            assert(field.len == self.manager.numNodes());
            assert(rhs.len == self.manager.numNodes());
            assert(src.len == self.manager.numNodes());

            for (0..self.manager.numBlocks()) |block_id| {
                self.residualBlock(block_id, field, rhs, operator, src);
            }
        }

        /// Computes the residual of a given operation on this level, storing the result in field.
        pub fn residualLevel(self: @This(), level: usize, field: []f64, rhs: []const f64, operator: anytype, src: []const f64) void {
            assert(field.len == self.manager.numNodes());
            assert(src.len == self.manager.numNodes());
            assert(rhs.len == self.manager.numNodes());

            const blocks = self.manager.level_to_blocks.range(level);
            for (blocks.start..blocks.end) |block_id| {
                self.residualBlock(block_id, field, rhs, operator, src);
            }
        }

        fn residualBlock(self: @This(), block_id: usize, field: []f64, rhs: []const f64, operator: anytype, src: []const f64) void {
            const Oper: type = @TypeOf(operator);

            checkOperator(N, M, Oper);

            const block_field: []f64 = self.manager.blockNodes(block_id, field);
            const block_rhs: []const f64 = self.manager.blockNodes(block_id, rhs);
            const block_range = self.manager.block_to_nodes.range(block_id);

            const node_space = self.blockNodeSpace(block_id);

            var nodes = node_space.nodes(0);

            while (nodes.next()) |node| {
                const engine = Engine(N, M, Oper.order){
                    .space = node_space,
                    .node = node,
                    .range = block_range,
                };

                const rval = node_space.value(node, block_rhs);

                node_space.setValue(node, block_field, rval - operator.apply(engine, src));
            }
        }

        // *********************************
        // Tau correction ******************
        // *********************************

        /// Computes the tau correction for this level, given a restricted residual, and a source function.
        /// Only fills field for non-leaf cells.
        pub fn tauCorrectLevel(self: @This(), level: usize, field: []f64, res: []const f64, operator: anytype, src: []const f64) void {
            assert(field.len == self.manager.numNodes());
            assert(src.len == self.manager.numNodes());
            assert(res.len == self.manager.numNodes());

            if (level == 0) {
                return;
            }

            const blocks = self.manager.level_to_blocks.range(level);
            for (blocks.start..blocks.end) |block_id| {
                self.tauCorrectBlock(block_id, field, res, operator, src);
            }
        }

        fn tauCorrectBlock(self: @This(), block_id: usize, field: []f64, res: []const f64, operator: anytype, src: []const f64) void {
            const Oper: type = @TypeOf(operator);

            checkOperator(N, M, Oper);

            assert(block_id != 0);

            const parent = self.manager.blocks.items(.parent)[block_id];
            const parent_field = self.manager.blockNodes(parent, field);
            const parent_res = self.manager.blockNodes(parent, res);
            const parent_range = self.manager.block_to_nodes.range(parent);
            const parent_node_space = self.blockNodeSpace(parent);

            const origin = self.manager.blockNodeParentOrigin(block_id);
            const size = self.manager.blockNodeParentSize(block_id);

            const indices = IndexSpace.fromSize(size).cartesianIndices();

            while (indices.next()) |offset| {
                const node = toSigned(IndexMixin.add(origin, offset));

                const engine = Engine(N, M, Oper.order){
                    .space = parent_node_space,
                    .node = node,
                    .range = parent_range,
                };

                const aval = operator.apply(engine, src);
                const rval = parent_node_space.value(node, parent_res);

                parent_node_space.setValue(node, parent_field, rval + aval);
            }
        }

        // ***************************************
        // Smoothing *****************************
        // ***************************************

        /// Performs jacobi smoothing on this level.
        pub fn smoothLevel(self: @This(), level: usize, field: []f64, operator: anytype, src: []const f64, rhs: []const f64) void {
            assert(field.len == self.manager.numNodes());
            assert(src.len == self.manager.numNodes());

            const blocks = self.manager.level_to_blocks.range(level);
            for (blocks.start..blocks.end) |block_id| {
                self.smoothBlock(block_id, field, operator, src, rhs);
            }
        }

        /// Performs jacobi smoothing on this block.
        fn smoothBlock(self: @This(), block_id: usize, field: []f64, operator: anytype, src: []const f64, rhs: []const f64) void {
            const Oper: type = @TypeOf(operator);

            checkOperator(N, M, Oper);

            const block_field: []f64 = self.manager.blockNodes(block_id, field);
            const block_rhs: []const f64 = self.manager.blockNodes(block_id, rhs);
            const block_src: []const f64 = self.manager.blockNodes(block_id, src);
            const block_range = self.manager.block_to_nodes.range(block_id);

            const node_space = self.blockNodeSpace(block_id);

            var nodes = node_space.nodes(0);

            while (nodes.next()) |node| {
                const engine = Engine{
                    .space = node_space,
                    .node = node,
                    .range = block_range,
                };

                const sval = node_space.value(node, block_src);
                const rval = node_space.value(node, block_rhs);
                const app = operator.apply(engine, src);
                const appDiag = operator.applyDiag(engine);

                node_space.setValue(node, block_field, sval + 2.0 / 3.0 * (rval - app) / appDiag);
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

                // **************************
                // Ghost Nodes **************
                // **************************

                /// Fills all ghost nodes.
                pub fn fillGhostNodes(self: @This(), set: anytype, field: []f64) void {
                    checkBoundarySet(N, @TypeOf(set));

                    const manager = self.worker.manager;
                    assert(field.len == manager.numNodes());

                    self.fillBaseGhostNodes(set, field);

                    for (1..manager.numBlocks()) |block_id| {
                        self.fillBlockGhostNodes(block_id, set, field);
                    }
                }

                /// Fills the ghost nodes of given level.
                pub fn fillLevelGhostNodes(self: @This(), level: usize, set: anytype, field: []f64) void {
                    checkBoundarySet(N, @TypeOf(set));

                    const manager = self.worker.manager;
                    assert(field.len == manager.numNodes());

                    // Short circuit in the case of level = 0
                    if (level == 0) {
                        self.fillBaseGhostNodes(set, field);
                    } else {
                        // Otherwise check each block on the level
                        const blocks = manager.level_to_blocks.range(level);
                        for (blocks.start..blocks.end) |block_id| {
                            self.fillBlockGhostNodes(block_id, set, field);
                        }
                    }
                }

                fn fillBaseGhostNodes(self: @This(), set: anytype, field: []f64) void {
                    const manager = self.worker.manager;
                    const node_space = self.worker.blockNodeSpace(0);

                    const boundary_engine = BoundaryEngine(N, M, @TypeOf(set)){
                        .space = node_space,
                        .range = manager.block_to_nodes.range(0),
                        .set = set,
                    };

                    boundary_engine.fill(O, field);
                }

                /// Fills the boundary of a given block, whether by exptrapolation or prolongation.
                fn fillBlockGhostNodes(self: @This(), block_id: usize, set: anytype, field: []f64) void {
                    const manager = self.worker.manager;

                    assert(block_id != 0);

                    const block_boundary = manager.blocks.items(.boundary)[block_id];
                    const node_space = self.worker.blockNodeSpace(block_id);

                    const boundary_engine = BoundaryEngine(N, M, @TypeOf(set)){
                        .space = node_space,
                        .range = manager.block_to_nodes.range(block_id),
                        .set = set,
                    };

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

                            bmask.setValue(axis, block_boundary[face.toLinear()]);
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
                                boundary_engine.fillRegion(O, region, imask, field);
                            }
                        }
                    }
                }

                /// Fills ghost nodes on the interior of the numerical domain.
                fn fillBlockGhostNodesInterior(self: @This(), region: Region, block_id: usize, field: []f64) void {
                    const manager = self.worker.manager;
                    const mesh = self.worker.mesh;

                    assert(block_id != 0);

                    const block_size = manager.blocks.items(.size)[block_id];
                    const block_node_space = self.worker.blockNodeSpace(block_id);
                    const block_field: []f64 = manager.blockNodes(block_id, field);

                    // Cache pointers
                    const cells = mesh.cells.slice();
                    const cell_meta = manager.cells.slice();
                    const cell_size = manager.cell_size;

                    var cell_indices = region.innerFaceCells(block_size);

                    while (cell_indices.next()) |index| {
                        // Uses permutation to find actual cell
                        const cell = manager.cellFromBlock(block_id, index);
                        // Origin in node space
                        const origin = toSigned(manager.cellNodeOrigin(cell));
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
                            const neighbor_index = cell_meta.items(.index)[neighbor];
                            const neighbor_field: []const f64 = manager.blockNodes(neighbor_block_id, field);
                            const neighbor_node_space = self.worker.blockNodeSpace(neighbor_block_id);

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
                            var offsets = region.nodes(O, manager.verticesPerCell());

                            while (offsets.next()) |offset| {
                                const node: [N]isize = addSigned(origin, offset);
                                const neighbor_node: [N]isize = addSigned(neighbor_origin, offset);

                                const v = neighbor_node_space.prolong(O, neighbor_node, neighbor_field);
                                block_node_space.setNodeValue(node, block_field, v);
                            }
                        } else {
                            // Neighbor is on same level.
                            const neighbor_block_id = cell_meta.items(.block)[neighbor];
                            const neighbor_index = cell_meta.items(.index)[neighbor];
                            const neighbor_field = manager.blockNodes(neighbor_block_id, field);
                            const neighbor_node_space = self.worker.blockNodeSpace(neighbor_block_id);

                            var neighbor_origin = toSigned(mul(neighbor_index, cell_size));

                            for (0..N) |axis| {
                                switch (region.sides[axis]) {
                                    .left => neighbor_origin[axis] += @intCast(cell_size[axis]),
                                    .right => neighbor_origin[axis] -= @intCast(cell_size[axis]),
                                    else => {},
                                }
                            }

                            var offsets = region.nodes(O, manager.verticesPerCell());

                            while (offsets.next()) |offset| {
                                const node = addSigned(origin, offset);
                                const neighbor_node = addSigned(neighbor_origin, offset);

                                const v = neighbor_node_space.nodeValue(neighbor_node, neighbor_field);
                                block_node_space.setNodeValue(node, block_field, v);
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

                    assert(block_id != 0);

                    // Cache pointers
                    const cell_size = manager.cell_size;

                    const block_field = manager.blockNodes(block_id, field);
                    const block_node_space = self.worker.blockNodeSpace(block_id);

                    const parent = manager.blocks.items(.parent)[block_id];
                    const parent_field = manager.blockNodes(parent, field);
                    const parent_node_space = self.worker.blockNodeSpace(parent);
                    const origin = refined(mul(manager.blocks.items(.index)[block_id], cell_size));

                    var node_indices = block_node_space.nodes(0);

                    while (node_indices.next()) |offset| {
                        const node = addSigned(origin, offset);

                        const val = parent_node_space.prolong(O, node, parent_field);
                        block_node_space.setNodeValue(node, block_field, val);
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

                    assert(block_id != 0);

                    // Cache pointers
                    const block_field = manager.blockNodes(block_id, field);
                    const block_node_space = self.worker.blockNodeSpace(block_id);

                    const parent = manager.blocks.items(.parent)[block_id];
                    const parent_field = manager.blockNodes(parent, field);
                    const parent_node_space = self.worker.blockNodeSpace(parent);

                    const origin = manager.blockNodeParentOrigin(block_id);
                    const size = manager.blockNodeParentSize(block_id);

                    const indices = IndexSpace.fromSize(size).cartesianIndices();

                    while (indices.next()) |offset| {
                        const node = toSigned(IndexMixin.add(origin, offset));

                        const val = block_node_space.restrictFull(toSigned(offset), block_field);
                        parent_node_space.setNodeValue(node, parent_field, val);
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
        fn blockNodeSpace(self: @This(), block_id: usize) NodeSpace {
            const block_bounds = self.manager.blocks.items(.bounds)[block_id];
            const block_size = self.manager.blockNodeSize(block_id);

            return .{
                .bounds = block_bounds,
                .size = block_size,
            };
        }
    };
}
