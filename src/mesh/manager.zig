const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const ArrayListUmanaged = std.ArrayListUnmanaged;
const MultiArrayList = std.MultiArrayList;
const assert = std.debug.assert;
const panic = std.debug.panic;

const mesh_ = @import("mesh.zig");
const null_index = mesh_.null_index;
const boundary_index = mesh_.boundary_index;

const permute = @import("permute.zig");

const common = @import("../common/common.zig");
const geometry = @import("../geometry/geometry.zig");
const utils = @import("../utils.zig");

const BoundaryEngine = common.BoundaryEngine;
const Engine = common.Engine;
const checkOperator = common.checkOperator;
const checkFunction = common.checkFunction;
const checkBoundarySet = common.checkBoundarySet;

const Range = utils.Range;
const RangeMap = utils.RangeMap;

/// Stores the additional structured needed to associate nodes with a mesh. Uniform children with common parents
/// are grouped into blocks to prevent unnessary ghost node duplication. Similar to a mesh, this node manager
/// describes the structure of a discretization.
///
/// All routines for transfering between node vectors, prolongation, smoothing, and filling ghost nodes are handled
/// by seperate `NodeWorker` classes which handle parallelism and dispatching work to other devices
/// (like the GPU or other processes).
pub fn NodeManager(comptime N: usize, comptime M: usize) type {
    return struct {
        /// Allocator for internal data structures
        gpa: Allocator,
        /// Base mesh on which nodes are located
        mesh: Mesh,
        /// Blocks defined on mesh
        blocks: BlockStructure,
        /// Map from blocks to node ranges
        block_to_nodes: RangeMap,
        /// Number of nodes per axis of each cell.
        cell_size: [N]usize,

        const Mesh = mesh_.Mesh(N);
        const BlockStructure = mesh_.BlockStructure(N);

        const NodeSpace = common.NodeSpace(N, M);

        const AxisMask = geometry.AxisMask(N);
        const FaceIndex = geometry.FaceIndex(N);
        const IndexSpace = geometry.IndexSpace(N);
        const IndexMixin = geometry.IndexMixin(N);
        const RealBox = geometry.RealBox(N);
        const Region = geometry.Region(N);
        const add = IndexMixin.add;
        const addSigned = IndexMixin.addSigned;
        const toSigned = IndexMixin.toSigned;

        pub fn init(allocator: Allocator, mesh: *const Mesh, cell_size: [N]usize, max_refinement: usize) !@This() {
            var cmesh = try mesh.clone(allocator);
            errdefer cmesh.deinit();

            var blocks = try BlockStructure.init(allocator, mesh, max_refinement);
            errdefer blocks.deinit(allocator);

            var block_to_nodes: RangeMap = .{};
            errdefer block_to_nodes.deinit(allocator);

            var offset: usize = 0;
            try block_to_nodes.append(allocator, offset);

            for (0..blocks.numBlocks()) |block_id| {
                const size = blocks.blockSize(block_id);

                var total: usize = 1;

                for (0..N) |axis| {
                    total *= cell_size[axis] * size[axis] + 1 + 2 * M;
                }

                offset += total;

                try block_to_nodes.append(allocator, offset);
            }

            return .{
                .gpa = allocator,
                .mesh = cmesh,
                .blocks = blocks,
                .block_to_nodes = block_to_nodes,
                .cell_size = cell_size,
            };
        }

        pub fn deinit(self: *@This()) void {
            self.block_to_nodes.deinit(self.gpa);
            self.blocks.deinit(self.gpa);
            self.mesh.deinit();
        }

        pub fn numLevels(self: *const @This()) usize {
            return self.mesh.numLevels();
        }

        pub fn numCells(self: *const @This()) usize {
            return self.mesh.numCells();
        }

        pub fn numBlocks(self: *const @This()) usize {
            return self.blocks.numBlocks();
        }

        pub fn numNodes(self: *const @This()) usize {
            return self.block_to_nodes.total();
        }

        /// Retrieves the spatial spacing of a given level
        pub fn spacing(self: *const @This(), level: usize) [N]f64 {
            var result: [N]f64 = self.mesh.physicalBounds();

            for (0..N) |axis| {
                result[axis] /= @floatFromInt(self.cell_size[axis]);
            }

            for (0..level) |_| {
                for (0..N) |axis| {
                    result[axis] /= 2.0;
                }
            }

            return result;
        }

        pub fn minSpacing(self: *const @This()) f64 {
            const sp = self.spacing(self.mesh.numLevels() - 1);

            var result = sp[0];

            for (1..N) |axis| {
                result = @max(result, sp[axis]);
            }

            return result;
        }

        /// Returns the number of vertices per cell along each axis, including vertices which overlap with other cells.
        pub fn verticesPerCell(self: *const @This()) [N]usize {
            var result = self.cell_size;

            for (0..N) |i| {
                result[i] += 1;
            }

            return result;
        }

        /// Returns a slice of nodes for the given block
        pub fn blockNodes(self: *const @This(), block_id: usize, data: anytype) @TypeOf(data) {
            return self.block_to_nodes.slice(block_id, data);
        }

        pub fn blockNodeRange(self: *const @This(), block_id: usize) Range {
            return self.block_to_nodes.range(block_id);
        }

        /// Returns the number of nodes
        pub fn blockNodeSize(self: *const @This(), block_id: usize) [N]usize {
            const size = self.blocks.blockSize(block_id);

            var result: [N]usize = undefined;

            for (0..N) |axis| {
                result[axis] = self.cell_size[axis] * size[axis] + 1;
            }

            return result;
        }

        /// Helper function for determining the node space of a block
        pub fn blockNodeSpace(self: @This(), block_id: usize) NodeSpace {
            const block_bounds = self.blocks.blockBounds(block_id);
            const block_size = self.blockNodeSize(block_id);

            return .{
                .bounds = block_bounds,
                .size = block_size,
            };
        }

        pub fn blockNodeParentOrigin(self: *const @This(), block_id: usize) [N]usize {
            var origin = self.blocks.blockParentIndex(block_id);

            for (0..N) |axis| {
                origin[axis] *= self.cell_size[axis];
            }

            return origin;
        }

        pub fn blockNodeParentSize(self: *const @This(), block_id: usize) [N]usize {
            const size = self.blocks.blockSize(block_id);

            var result: [N]usize = undefined;

            for (0..N) |axis| {
                result[axis] = (self.cell_size[axis] * size[axis]) / 2 + 1;
            }

            return result;
        }

        pub fn cellNodeOrigin(self: *const @This(), cell_id: usize) [N]usize {
            var origin = self.blocks.cellBlockIndex(cell_id);

            for (0..N) |axis| {
                origin[axis] *= self.cell_size[axis];
            }

            return origin;
        }

        // *************************************************
        // Kernels *****************************************
        // *************************************************

        /// Computes the l2 norm of the the field, including nodes on non-leaf cells.
        pub fn norm(self: @This(), field: []const f64) f64 {
            assert(field.len == self.numNodes());

            var result: f64 = 0.0;

            for (0..self.numBlocks()) |block_id| {
                const block_field = self.blockNodes(block_id, field);
                const node_space = self.blockNodeSpace(block_id);

                var nodes = node_space.nodes(0);

                while (nodes.next()) |node| {
                    const v = node_space.value(node, block_field);
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
            assert(src.len == self.numNodes());
            assert(dest.len == self.numNodes());

            @memcpy(dest, src);
        }

        /// Copies src to dest on a particular level.
        pub fn copyLevel(self: @This(), level: usize, dest: []f64, src: []const f64) void {
            assert(src.len == self.numNodes());
            assert(dest.len == self.numNodes());

            const blocks = self.blocks.levelBlocks(level);

            for (blocks.start..blocks.end) |block_id| {
                const block_src = self.blockNodes(block_id, src);
                const block_dest = self.blockNodes(block_id, dest);

                @memcpy(block_dest, block_src);
            }
        }

        // /// Adds v to dest on a particular level.
        // pub fn addAssignLevel(self: @This(), level: usize, dest: []f64, v: []const f64) void {
        //     assert(v.len == self.numNodes());
        //     assert(dest.len == self.numNodes());

        //     const blocks = self.blocks.levelBlocks(level);

        //     for (blocks.start..blocks.end) |block_id| {
        //         const block_v = self.blockNodes(block_id, v);
        //         const block_dest = self.blockNodes(block_id, dest);

        //         for (0..block_dest.len) |i| {
        //             block_dest[i] += block_v[i];
        //         }
        //     }
        // }

        // /// Subtracts v from dest on a particular level.
        // pub fn subAssignLevel(self: @This(), level: usize, dest: []f64, v: []const f64) void {
        //     assert(v.len == self.numNodes());
        //     assert(dest.len == self.numNodes());

        //     const blocks = self.blocks.levelBlocks(level);

        //     for (blocks.start..blocks.end) |block_id| {
        //         const block_v = self.blockNodes(block_id, v);
        //         const block_dest = self.blockNodes(block_id, dest);

        //         for (0..block_dest.len) |i| {
        //             block_dest[i] -= block_v[i];
        //         }
        //     }
        // }

        // *********************************
        // Base operations *****************
        // *********************************

        pub fn numBaseNodes(self: @This()) usize {
            return IndexSpace.fromSize(self.verticesPerCell()).total();
        }

        pub fn packBase(self: @This(), dest: []f64, src: []const f64) void {
            assert(dest.len == self.numBaseNodes());
            assert(src.len == self.numNodes());

            const block_src = self.blockNodes(0, src);
            const node_space = self.blockNodeSpace(0);

            var vertices = node_space.nodes(0);
            var linear: usize = 0;

            while (vertices.next()) |vertex| : (linear += 1) {
                const v = node_space.value(vertex, block_src);
                dest[linear] = v;
            }
        }

        pub fn unpackBase(self: @This(), dest: []f64, src: []const f64) void {
            assert(src.len == self.numBaseNodes());
            assert(dest.len == self.numNodes());

            const block_dest = self.blockNodes(0, dest);
            const node_space = self.blockNodeSpace(0);

            var vertices = node_space.nodes(0);
            var linear: usize = 0;

            while (vertices.next()) |vertex| : (linear += 1) {
                node_space.setValue(vertex, block_dest, src[linear]);
            }
        }

        // **************************************
        // Refinement Helpers *******************
        // **************************************

        /// Computse refinement flags based on local truncation error. If the absolute value of the error for any node in a cell
        /// is greater than the given tolerance, that cell will be tagged for refinement.
        pub fn refineByLocalTruncErr(self: @This(), err: []const f64, tol: f64, flags: []bool) void {
            assert(flags.len == self.numCells());
            assert(err.len == self.numNodes());
            assert(tol > 0);

            for (0..self.numCells()) |cell_id| {
                const block_id = self.blocks.cellBlock(cell_id);
                const origin = self.cellNodeOrigin(cell_id);

                const block_err = self.blockNodes(block_id, err);
                const node_space = self.blockNodeSpace(block_id);

                var max_err: f64 = 0.0;

                var offsets = IndexSpace.fromSize(self.verticesPerCell()).cartesianIndices();

                while (offsets.next()) |offset| {
                    const node = toSigned(add(origin, offset));

                    const v = node_space.value(node, block_err);
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
            for (0..self.numLevels()) |level_id| {
                self.projectLevel(level_id, projection, field);
            }
        }

        /// Using the given projection to set the values of the field on this level.
        pub fn projectLevel(self: @This(), level: usize, projection: anytype, field: []f64) void {
            assert(field.len == self.numNodes());

            const blocks = self.blocks.levelBlocks(level);

            for (blocks.start..blocks.end) |block_id| {
                self.projectBlock(block_id, projection, field);
            }
        }

        /// Set values of the field on the given block using the projection.
        fn projectBlock(self: @This(), block_id: usize, projection: anytype, field: []f64) void {
            const Proj: type = @TypeOf(projection);

            checkFunction(N, M, Proj);

            const block_field: []f64 = self.blockNodes(block_id, field);
            const block_range = self.blockNodeRange(block_id);

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
            assert(field.len == self.numNodes());
            assert(src.len == self.numNodes());

            for (0..self.numBlocks()) |block_id| {
                self.applyBlock(block_id, field, operator, src);
            }
        }

        /// Applies the operator to the source function on this level, storing the result in field.
        pub fn applyLevel(self: @This(), level: usize, field: []f64, operator: anytype, src: []const f64) void {
            assert(field.len == self.numNodes());
            assert(src.len == self.numNodes());

            const blocks = self.blocks.levelBlocks(level);
            for (blocks.start..blocks.end) |block_id| {
                self.applyBlock(block_id, field, operator, src);
            }
        }

        /// Applies the operator to the source function on this block, storing the result on field.
        fn applyBlock(self: @This(), block_id: usize, field: []f64, operator: anytype, src: []const f64) void {
            const Oper: type = @TypeOf(operator);

            checkOperator(N, M, Oper);

            const block_field: []f64 = self.blockNodes(block_id, field);
            const block_range = self.blockNodeRange(block_id);

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
            assert(field.len == self.numNodes());
            assert(rhs.len == self.numNodes());
            assert(src.len == self.numNodes());

            for (0..self.numBlocks()) |block_id| {
                self.residualBlock(block_id, field, rhs, operator, src);
            }
        }

        /// Computes the residual of a given operation on this level, storing the result in field.
        pub fn residualLevel(self: @This(), level: usize, field: []f64, rhs: []const f64, operator: anytype, src: []const f64) void {
            assert(field.len == self.numNodes());
            assert(src.len == self.numNodes());
            assert(rhs.len == self.numNodes());

            const blocks = self.blocks.levelBlocks(level);

            for (blocks.start..blocks.end) |block_id| {
                self.residualBlock(block_id, field, rhs, operator, src);
            }
        }

        fn residualBlock(self: @This(), block_id: usize, field: []f64, rhs: []const f64, operator: anytype, src: []const f64) void {
            const Oper: type = @TypeOf(operator);

            checkOperator(N, M, Oper);

            const block_field: []f64 = self.blockNodes(block_id, field);
            const block_rhs: []const f64 = self.blockNodes(block_id, rhs);
            const block_range = self.blockNodeRange(block_id);

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

        // ***************************************
        // Smoothing *****************************
        // ***************************************

        /// Performs jacobi smoothing on this level.
        pub fn smoothLevel(self: @This(), level: usize, field: []f64, operator: anytype, src: []const f64, rhs: []const f64) void {
            assert(field.len == self.numNodes());
            assert(src.len == self.numNodes());

            const blocks = self.blocks.levelBlocks(level);

            for (blocks.start..blocks.end) |block_id| {
                self.smoothBlock(block_id, field, operator, src, rhs);
            }
        }

        /// Performs Jacobi smoothing on this block.
        fn smoothBlock(self: @This(), block_id: usize, field: []f64, operator: anytype, src: []const f64, rhs: []const f64) void {
            const Oper: type = @TypeOf(operator);

            checkOperator(N, M, Oper);

            const block_field: []f64 = self.blockNodes(block_id, field);
            const block_rhs: []const f64 = self.blockNodes(block_id, rhs);
            const block_src: []const f64 = self.blockNodes(block_id, src);
            const block_range = self.blockNodeRange(block_id);

            const node_space = self.blockNodeSpace(block_id);

            var nodes = node_space.nodes(0);

            while (nodes.next()) |node| {
                const engine = Engine(N, M, Oper.order){
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

        // **********************************
        // Dissipation **********************
        // **********************************

        /// Applies dissipation to a first order hyperbolic system of equations.
        /// Here stencils have been scaled such that, in order to perserve stability,
        /// `eps` must be less than or equal to `min(delta_x)/(delta_t)`, which may be written
        /// `eps <= 1/max(CFL)`.
        pub fn dissipation(self: @This(), comptime O: usize, eps: f64, deriv: []f64, src: []const f64) void {
            assert(deriv.len == self.numNodes());
            assert(src.len == self.numNodes());

            for (0..self.numBlocks()) |block_id| {
                self.dissipationBlock(O, block_id, eps, deriv, src);
            }
        }

        pub fn dissipationLevel(self: @This(), comptime O: usize, level: usize, eps: f64, deriv: []f64, src: []const f64) void {
            assert(deriv.len == self.numNodes());
            assert(src.len == self.numNodes());

            const blocks = self.blocks.levelBlocks(level);
            for (blocks.start..blocks.end) |block_id| {
                self.dissipationBlock(O, block_id, eps, deriv, src);
            }
        }

        fn dissipationBlock(self: @This(), comptime O: usize, block_id: usize, eps: f64, deriv: []f64, src: []const f64) void {
            const block_deriv: []f64 = self.blockNodes(block_id, deriv);
            const block_src: []const f64 = self.blockNodes(block_id, src);

            const block_node_space = self.blockNodeSpace(block_id);

            const spaces = block_node_space.spacing();

            var nodes = block_node_space.nodes(0);

            while (nodes.next()) |node| {
                const v = block_node_space.value(node, block_deriv);

                var result: f64 = 0.0;

                inline for (0..N) |axis| {
                    result += eps / spaces[axis] * block_node_space.dissipationAxis(O, axis, node, block_src);
                }

                block_node_space.setValue(node, block_deriv, v + result);
            }
        }

        // *********************************
        // Tau correction ******************
        // *********************************

        /// Computes the tau correction for this level, given a restricted residual, and a source function.
        /// Only fills field for non-leaf cells.
        pub fn tauCorrectLevel(self: @This(), level: usize, field: []f64, res: []const f64, operator: anytype, src: []const f64) void {
            assert(field.len == self.numNodes());
            assert(src.len == self.numNodes());
            assert(res.len == self.numNodes());

            if (level == 0) {
                return;
            }

            const blocks = self.blocks.levelBlocks(level);

            for (blocks.start..blocks.end) |block_id| {
                self.tauCorrectBlock(block_id, field, res, operator, src);
            }
        }

        fn tauCorrectBlock(self: @This(), block_id: usize, field: []f64, res: []const f64, operator: anytype, src: []const f64) void {
            const Oper: type = @TypeOf(operator);

            checkOperator(N, M, Oper);

            assert(block_id != 0);

            const parent = self.blocks.blockParent(block_id);
            const parent_field = self.blockNodes(parent, field);
            const parent_res = self.blockNodes(parent, res);
            const parent_range = self.blockNodeRange(parent);
            const parent_node_space = self.blockNodeSpace(parent);

            const origin = self.blockNodeParentOrigin(block_id);
            const size = self.blockNodeParentSize(block_id);

            var indices = IndexSpace.fromSize(size).cartesianIndices();

            while (indices.next()) |offset| {
                const node = toSigned(add(origin, offset));

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

        pub fn sigmaCorrectLevel(self: @This(), comptime O: usize, level: usize, field: []f64, new: []const f64, old: []const f64) void {
            assert(field.len == self.numNodes());
            assert(new.len == self.numNodes());
            assert(old.len == self.numNodes());

            if (level == 0) {
                return;
            }

            const blocks = self.blocks.levelBlocks(level);

            for (blocks.start..blocks.end) |block_id| {
                self.sigmaCorrectBlock(O, block_id, field, new, old);
            }
        }

        fn sigmaCorrectBlock(self: @This(), comptime O: usize, block_id: usize, field: []f64, new: []const f64, old: []const f64) void {
            assert(block_id != 0);

            const block_field = self.blockNodes(block_id, field);
            const block_node_space = self.blockNodeSpace(block_id);

            const parent = self.blocks.blockParent(block_id);
            const parent_new = self.blockNodes(parent, new);
            const parent_old = self.blockNodes(parent, old);

            const parent_node_space = self.blockNodeSpace(parent);

            std.debug.print("Num Nodes {}, Slice {}\n", .{ parent_node_space.numNodes(), parent_new.len });

            const origin = toSigned(IndexMixin.refined(self.blockNodeParentOrigin(block_id)));

            var node_indices = block_node_space.nodes(0);

            while (node_indices.next()) |offset| {
                const node = addSigned(origin, offset);

                const pnew = parent_node_space.prolong(O, node, parent_new);
                const pold = parent_node_space.prolong(O, node, parent_old);

                const val = block_node_space.value(offset, block_field);
                block_node_space.setValue(offset, block_field, val + pnew - pold);
            }
        }

        // **************************************
        // Restriction/Prolongation *************
        // **************************************

        /// Prolongs data from `level - 1` to `level`.
        pub fn prolongLevel(self: @This(), comptime O: usize, level: usize, field: []f64) void {
            assert(field.len == self.numNodes());

            if (level == 0) {
                return;
            }

            const blocks = self.blocks.levelBlocks(level);

            for (blocks.start..blocks.end) |block_id| {
                self.prolongBlock(O, block_id, field);
            }
        }

        fn prolongBlock(self: @This(), comptime O: usize, block_id: usize, field: []f64) void {
            assert(block_id != 0);

            const block_field = self.blockNodes(block_id, field);
            const block_node_space = self.blockNodeSpace(block_id);

            const parent = self.blocks.blockParent(block_id);
            const parent_field = self.blockNodes(parent, field);
            const parent_node_space = self.blockNodeSpace(parent);

            const origin = toSigned(IndexMixin.refined(self.blockNodeParentOrigin(block_id)));

            var node_indices = block_node_space.nodes(0);

            while (node_indices.next()) |offset| {
                const node = addSigned(origin, offset);

                const val = parent_node_space.prolong(O, node, parent_field);
                block_node_space.setValue(offset, block_field, val);
            }
        }

        /// Restricts data from `level` to `level - 1`
        pub fn restrictLevel(self: @This(), comptime O: usize, level: usize, field: []f64) void {
            assert(field.len == self.numNodes());

            if (level == 0) {
                return;
            }

            const blocks = self.blocks.levelBlocks(level);

            for (blocks.start..blocks.end) |block_id| {
                self.restrictBlock(O, block_id, field);
            }
        }

        fn restrictBlock(self: @This(), comptime O: usize, block_id: usize, field: []f64) void {
            assert(block_id != 0);

            // Cache pointers
            const block_field = self.blockNodes(block_id, field);
            const block_node_space = self.blockNodeSpace(block_id);

            const parent = self.blocks.blockParent(block_id);
            const parent_field = self.blockNodes(parent, field);
            const parent_node_space = self.blockNodeSpace(parent);

            const origin = toSigned(self.blockNodeParentOrigin(block_id));

            const size = self.blockNodeParentSize(block_id);

            var offsets = IndexSpace.fromSize(size).cartesianIndices();

            while (offsets.next()) |offset| {
                const soffset = toSigned(offset);
                const node = addSigned(origin, soffset);

                const val = block_node_space.restrict(O, soffset, block_field);
                parent_node_space.setValue(node, parent_field, val);
            }
        }

        // **************************
        // Ghost Nodes **************
        // **************************

        /// Fills all ghost nodes.
        pub fn fillGhostNodes(self: @This(), comptime O: usize, set: anytype, field: []f64) void {
            assert(field.len == self.numNodes());

            self.fillBaseGhostNodes(O, set, field);

            for (1..self.numBlocks()) |block_id| {
                self.fillBlockGhostNodes(O, block_id, set, field);
            }
        }

        /// Fills the ghost nodes of given level.
        pub fn fillLevelGhostNodes(self: @This(), comptime O: usize, level: usize, set: anytype, field: []f64) void {
            assert(field.len == self.numNodes());

            // Short circuit in the case of level = 0
            if (level == 0) {
                self.fillBaseGhostNodes(O, set, field);
            } else {
                // Otherwise check each block on the level
                const blocks = self.blocks.levelBlocks(level);
                for (blocks.start..blocks.end) |block_id| {
                    self.fillBlockGhostNodes(O, block_id, set, field);
                }
            }
        }

        fn fillBaseGhostNodes(self: @This(), comptime O: usize, set: anytype, field: []f64) void {
            checkBoundarySet(N, @TypeOf(set));

            const node_space = self.blockNodeSpace(0);

            const boundary_engine = BoundaryEngine(N, M, @TypeOf(set)){
                .space = node_space,
                .range = self.blockNodeRange(0),
                .set = set,
            };

            boundary_engine.fill(O, field);
        }

        /// Fills the boundary of a given block, whether by exptrapolation or prolongation.
        fn fillBlockGhostNodes(self: @This(), comptime O: usize, block_id: usize, set: anytype, field: []f64) void {
            checkBoundarySet(N, @TypeOf(set));

            assert(block_id != 0);

            const block_boundary = self.blocks.blockBoundary(block_id);
            const node_space = self.blockNodeSpace(block_id);

            const boundary_engine = BoundaryEngine(N, M, @TypeOf(set)){
                .space = node_space,
                .range = self.blockNodeRange(block_id),
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
                    self.fillBlockGhostNodesInterior(O, region, block_id, field);
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
        fn fillBlockGhostNodesInterior(self: @This(), comptime O: usize, region: Region, block_id: usize, field: []f64) void {
            assert(block_id != 0);

            const block_size = self.blocks.blockSize(block_id);
            const block_node_space = self.blockNodeSpace(block_id);
            const block_field: []f64 = self.blockNodes(block_id, field);

            // Cache pointers
            const cells = self.mesh.cells.slice();
            const cell_meta = self.blocks.cells.slice();
            const cell_size = self.cell_size;

            var cell_indices = region.innerFaceCells(block_size);

            while (cell_indices.next()) |index| {
                // Uses permutation to find actual cell
                const cell = self.blocks.blockCell(block_id, index);
                // Origin in node space
                const origin = toSigned(self.cellNodeOrigin(cell));
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
                    const neighbor_field: []const f64 = self.blockNodes(neighbor_block_id, field);
                    const neighbor_node_space = self.blockNodeSpace(neighbor_block_id);

                    var neighbor_origin = toSigned(IndexMixin.refined(IndexMixin.mul(neighbor_index, cell_size)));

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
                    var offsets = region.nodes(M, self.verticesPerCell());

                    while (offsets.next()) |offset| {
                        const node: [N]isize = addSigned(origin, offset);
                        const neighbor_node: [N]isize = addSigned(neighbor_origin, offset);

                        const v = neighbor_node_space.prolong(O, neighbor_node, neighbor_field);
                        block_node_space.setValue(node, block_field, v);
                    }
                } else {
                    // Neighbor is on same level.
                    const neighbor_block_id = cell_meta.items(.block)[neighbor];
                    const neighbor_index = cell_meta.items(.index)[neighbor];
                    const neighbor_field = self.blockNodes(neighbor_block_id, field);
                    const neighbor_node_space = self.blockNodeSpace(neighbor_block_id);

                    var neighbor_origin = toSigned(IndexMixin.mul(neighbor_index, cell_size));

                    for (0..N) |axis| {
                        switch (region.sides[axis]) {
                            .left => neighbor_origin[axis] += @intCast(cell_size[axis]),
                            .right => neighbor_origin[axis] -= @intCast(cell_size[axis]),
                            else => {},
                        }
                    }

                    var offsets = region.nodes(M, self.verticesPerCell());

                    while (offsets.next()) |offset| {
                        const node = addSigned(origin, offset);
                        const neighbor_node = addSigned(neighbor_origin, offset);

                        const v = neighbor_node_space.value(neighbor_node, neighbor_field);
                        block_node_space.setValue(node, block_field, v);
                    }
                }
            }
        }
    };
}
