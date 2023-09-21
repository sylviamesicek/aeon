// std imports
const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const ArrayListUnmanaged = std.ArrayListUnmanaged;
const MultiArrayList = std.MultiArrayList;
const ArenaAllocator = std.heap.ArenaAllocator;
const assert = std.debug.assert;
const exp2 = std.math.exp2;
const maxInt = std.math.maxInt;

// root imports

const basis = @import("../basis/basis.zig");
const geometry = @import("../geometry/geometry.zig");
const solver = @import("../solver/solver.zig");
const system = @import("../system.zig");

// Submodules

const dofs = @import("dofs.zig");
const operator = @import("operator.zig");
const levels = @import("level.zig");

// Public Exports

pub const DofHandler = dofs.DofHandler;
pub const OperatorEngine = operator.OperatorEngine;
pub const FunctionEngine = operator.FunctionEngine;
pub const EngineType = operator.EngineType;
pub const isMeshOperator = operator.isMeshOperator;
pub const isMeshFunction = operator.isMeshFunction;

pub fn Mesh(comptime N: usize, comptime O: usize) type {
    return struct {
        /// Allocator used for various arraylists stored in this struct.
        gpa: Allocator,
        /// Configuration of Mesh
        config: Config,
        /// Total number of tiles in mesh
        tile_total: usize,
        /// Total number of cells in mesh
        cell_total: usize,
        /// Base level (after performing global_refinement).
        base: Base,
        /// Refined levels
        levels: ArrayListUnmanaged(Level),
        /// Number of levels which are active.
        active_levels: usize,

        // Public types
        pub const Level = levels.Level(N, O);
        pub const Block = levels.Block(N);
        pub const Patch = levels.Patch(N);

        pub const Config = struct {
            physical_bounds: RealBox,
            index_size: [N]usize,
            tile_width: usize,

            pub fn check(self: Config) void {
                assert(self.tile_width >= 1);
                assert(self.tile_width >= 2 * O);
                for (0..N) |i| {
                    assert(self.index_size[i] > 0);
                    assert(self.physical_bounds.size[i] > 0.0);
                }
            }

            fn baseTileSpace(self: Config) IndexSpace {
                return IndexSpace.fromSize(self.index_size);
            }

            fn baseCellSpace(self: Config) IndexSpace {
                return IndexSpace.fromSize(add(scaled(self.index_size, self.tile_width), splat(4 * O)));
            }
        };

        pub const Base = struct {
            index_size: [N]usize,
            tile_total: usize,
            cell_total: usize,
        };

        // Aliases
        const Self = @This();
        const IndexBox = geometry.Box(N, usize);
        const RealBox = geometry.Box(N, f64);
        const Face = geometry.Face(N);
        const IndexSpace = geometry.IndexSpace(N);
        const PartitionSpace = geometry.PartitionSpace(N);
        const Region = geometry.Region(N);
        const CellSpace = basis.CellSpace(N, O);
        const StencilSpace = basis.StencilSpace(N, O);

        // Mixins
        const Index = @import("../index.zig").Index(N);

        const add = Index.add;
        const sub = Index.sub;
        const scaled = Index.scaled;
        const splat = Index.splat;
        const toSigned = Index.toSigned;

        /// Initialises a new mesh with an general purpose allocator, subject to the given configuration.
        pub fn init(allocator: Allocator, config: Config) Self {
            // Check config
            config.check();
            // Scale initial size by 2^global_refinement
            const tile_space: IndexSpace = config.baseTileSpace();
            const cell_space: IndexSpace = config.baseCellSpace();

            const base: Base = .{
                .index_size = tile_space.size,
                .tile_total = tile_space.total(),
                .cell_total = cell_space.total(),
            };

            return .{
                .gpa = allocator,
                .config = config,
                .tile_total = base.tile_total,
                .cell_total = base.cell_total,
                .base = base,
                .levels = .{},
                .active_levels = 0,
            };
        }

        /// Deinitalises a mesh.
        pub fn deinit(self: *Self) void {
            for (self.levels.items) |*level| {
                level.deinit(self.gpa);
            }

            self.levels.deinit(self.gpa);
        }

        // **********************************
        // Helpers **************************
        // **********************************

        pub fn tileTotal(self: *const Self) usize {
            return self.tile_total;
        }

        pub fn cellTotal(self: *const Self) usize {
            return self.cell_total;
        }

        pub fn baseTileSlice(self: *const Self, mesh_slice: anytype) @TypeOf(mesh_slice) {
            return mesh_slice[0..self.base.tile_total];
        }

        pub fn baseCellSlice(self: *const Self, mesh_slice: anytype) @TypeOf(mesh_slice) {
            return mesh_slice[0..self.base.cell_total];
        }

        // *************************
        // Block Map ***************
        // *************************

        pub fn buildBlockMap(self: *const Self, map: []usize) !void {
            assert(map.len == self.tileTotal());

            @memset(map, maxInt(usize));
            @memset(self.baseTileSlice(usize, map), 0);

            for (self.levels.items, 0..) |*level, l| {
                const level_map: []usize = self.levelTileSlice(l, usize, map);

                for (level.blocks.items(.bounds), level.blocks.items(.patch), 0..) |bounds, parent, id| {
                    const pbounds: IndexBox = level.patches.items(.bounds)[parent];
                    const tile_offset: usize = level.patches.items(.tile_offset)[parent];
                    const tile_total: usize = level.patches.items(.tile_total)[parent];
                    const tile_to_block: []usize = level_map[tile_offset..(tile_offset + tile_total)];

                    pbounds.space().fillSubspace(bounds.relativeTo(pbounds), usize, tile_to_block, id);
                }
            }
        }

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

        pub fn baseStencilSpace(self: *const Self) StencilSpace {
            return .{
                .physical_bounds = self.config.physical_bounds,
                .size = scaled(self.base.index_size, self.config.tile_width),
            };
        }

        pub fn levelStencilSpace(self: *const Self, level: usize, block: usize) StencilSpace {
            const bounds: IndexBox = self.levels.items[level].blocks.items(.bounds)[block];
            return .{
                .physical_bounds = self.blockPhysicalBounds(level, bounds),
                .size = scaled(bounds.size, self.config.tile_width),
            };
        }

        fn blockPhysicalBounds(self: *const Self, level: usize, block: IndexBox) RealBox {
            const index_size: [N]usize = self.levels.items[level].index_size;

            var physical_bounds: RealBox = undefined;

            for (0..N) |i| {
                const sratio: f64 = @as(f64, @floatFromInt(block.size[i])) / @as(f64, @floatFromInt(index_size[i]));
                const oratio: f64 = @as(f64, @floatFromInt(block.origin[i])) / @as(f64, @floatFromInt(index_size[i] - 1));

                physical_bounds.size[i] = self.config.physical_bounds.size[i] * sratio;
                physical_bounds.origin[i] = self.config.physical_bounds.origin[i] + self.config.physical_bounds.size[i] * oratio;
            }

            return physical_bounds;
        }

        // *************************
        // Smoothing ***************
        // *************************

        /// Runs one iteration of Jacobi's method on the full mesh. Assuming all boundaries have been filled at least partially.
        pub fn smooth(
            self: *const Self,
            oper: anytype,
            result: system.SystemSlice(@TypeOf(oper).System),
            input: system.SystemSliceConst(@TypeOf(oper).System),
            rhs: system.SystemSliceConst(@TypeOf(oper).System),
            context: system.SystemSliceConst(@TypeOf(oper).Context),
        ) void {
            self.smoothBase(oper, result, input, rhs, context);

            for (0..self.active_levels) |level| {
                self.smoothLevel(level, oper, result, input, rhs, context);
            }
        }

        /// Runs one iteration of Jacobi's method on the base level, assuming all boundaries have been filled at least partially.
        pub fn smoothBase(
            self: *const Self,
            oper: anytype,
            result: system.SystemSlice(@TypeOf(oper).System),
            input: system.SystemSliceConst(@TypeOf(oper).System),
            rhs: system.SystemSliceConst(@TypeOf(oper).System),
            context: system.SystemSliceConst(@TypeOf(oper).Context),
        ) void {
            if (!(operator.isMeshOperator(N, O)(@TypeOf(oper)))) {
                @compileError("Oper must satisfy isMeshOperator trait.");
            }

            const base_offset = 0;
            const base_total = self.base.cell_total;

            const base_result = system.systemStructSlice(result, base_offset, base_total);
            const base_input = system.systemStructSlice(input, base_offset, base_total);
            const base_rhs = system.systemStructSlice(rhs, base_offset, base_total);
            const base_context = system.systemStructSlice(context, base_offset, base_total);

            const stencil_space: StencilSpace = self.baseStencilSpace();

            var cells = stencil_space.cellSpace().cells();

            while (cells.next()) |cell| {
                const engine = EngineType(N, O, @TypeOf(oper)){
                    .inner = .{
                        .space = stencil_space,
                        .cell = cell,
                    },
                    .context = base_context,
                    .operated = base_input,
                };

                const app: system.SystemValue(@TypeOf(oper).System) = oper.apply(engine);
                const diag: system.SystemValue(@TypeOf(oper).System) = oper.applyDiagonal(engine);

                inline for (system.systemFieldNames(@TypeOf(oper).System)) |name| {
                    const f: f64 = stencil_space.value(cell, @field(base_input, name));
                    const r: f64 = stencil_space.value(cell, @field(base_rhs, name));
                    const a: f64 = @field(app, name);
                    const d: f64 = @field(diag, name);

                    stencil_space.setValue(cell, @field(base_result, name), f + (r - a) / d);
                }
            }
        }

        /// Runs one iteration of Jacobi's method on the given level, assuming all boundaries have been filled at least partially.
        pub fn smoothLevel(
            self: *const Self,
            level: usize,
            oper: anytype,
            result: system.SystemSlice(@TypeOf(oper).System),
            input: system.SystemSliceConst(@TypeOf(oper).System),
            rhs: system.SystemSliceConst(@TypeOf(oper).System),
            context: system.SystemSliceConst(@TypeOf(oper).Context),
        ) void {
            if (!(operator.isMeshOperator(N, O)(@TypeOf(oper)))) {
                @compileError("Oper must satisfy isMeshOperator trait.");
            }

            const target: *const Level = &self.levels[level];

            const level_offset = target.cell_offset;
            const level_total = target.cell_total;

            const level_result = system.systemStructSlice(result, level_offset, level_total);
            const level_input = system.systemStructSlice(input, level_offset, level_total);
            const level_rhs = system.systemStructSlice(rhs, level_offset, level_total);
            const level_context = system.systemStructSlice(context, level_offset, level_total);

            for (target.blocks.items(.bounds), target.blocks.items(.cell_offset), target.blocks.items(.cell_total)) |block, offset, total| {
                const block_result = system.systemStructSlice(level_result, offset, total);
                const block_input = system.systemStructSlice(level_input, offset, total);
                const block_rhs = system.systemStructSlice(level_rhs, offset, total);
                const block_context = system.systemStructSlice(level_context, offset, total);

                const stencil_space: StencilSpace = self.levelStencilSpace(level, block);

                var cells = stencil_space.cellSpace().cells();

                while (cells.next()) |cell| {
                    const engine = EngineType(N, O, @TypeOf(oper)){
                        .inner = .{
                            .space = stencil_space,
                            .cell = cell,
                        },
                        .context = block_context,
                        .operated = block_input,
                    };

                    const app: system.SystemValue(@TypeOf(oper).System) = oper.apply(engine);
                    const diag: system.SystemValue(@TypeOf(oper).System) = oper.applyDiagonal(engine);

                    inline for (system.systemFieldNames(@TypeOf(oper).System)) |name| {
                        const f: f64 = stencil_space.value(cell, @field(block_input, name));
                        const r: f64 = stencil_space.value(cell, @field(block_rhs, name));
                        const a: f64 = @field(app, name);
                        const d: f64 = @field(diag, name);

                        stencil_space.setValue(cell, @field(block_result, name), f + (r - a) / d);
                    }
                }
            }
        }

        // Dofs

        pub fn dofHandler(self: *const Self) DofHandler(N, O) {
            return .{
                .mesh = self,
            };
        }

        // *************************
        // Application *************
        // *************************

        pub fn apply(
            self: *const Self,
            comptime full: bool,
            oper: anytype,
            result: system.SystemSlice(@TypeOf(oper).System),
            input: system.SystemSliceConst(@TypeOf(oper).System),
            context: system.SystemSliceConst(@TypeOf(oper).Context),
        ) void {
            self.applyBase(oper, result, input, context);

            for (0..self.active_levels) |level| {
                self.applyLevel(full, level, result, input, context);
            }
        }

        pub fn applyBase(
            self: *const Self,
            oper: anytype,
            result: system.SystemSlice(@TypeOf(oper).System),
            input: system.SystemSliceConst(@TypeOf(oper).System),
            context: system.SystemSliceConst(@TypeOf(oper).Context),
        ) void {
            if (!(operator.isMeshOperator(N, O)(@TypeOf(oper)))) {
                @compileError("Oper must satisfy isMeshOperator trait.");
            }

            const base_offset = 0;
            const base_total = self.base.cell_total;

            const base_result = system.systemStructSlice(result, base_offset, base_total);
            const base_input = system.systemStructSlice(input, base_offset, base_total);
            const base_context = system.systemStructSlice(context, base_offset, base_total);

            const stencil_space: StencilSpace = self.baseStencilSpace();

            var cells = stencil_space.cellSpace().cells();

            while (cells.next()) |cell| {
                const engine = EngineType(N, O, @TypeOf(oper)){
                    .inner = .{
                        .space = stencil_space,
                        .cell = cell,
                    },
                    .context = base_context,
                    .operated = base_input,
                };

                const app: system.SystemValue(@TypeOf(oper).System) = oper.apply(engine);

                inline for (system.systemFieldNames(@TypeOf(oper).System)) |name| {
                    stencil_space.setValue(
                        cell,
                        @field(base_result, name),
                        @field(app, name),
                    );
                }
            }
        }

        pub fn applyLevel(
            self: *const Self,
            comptime full: bool,
            level: usize,
            oper: anytype,
            result: system.SystemSlice(@TypeOf(oper).System),
            input: system.SystemSliceConst(@TypeOf(oper).System),
            context: system.SystemSliceConst(@TypeOf(oper).Context),
        ) void {
            if (!(operator.isMeshOperator(N, O)(@TypeOf(oper)))) {
                @compileError("Oper must satisfy isMeshOperator trait.");
            }

            const target: *const Level = &self.levels[level];

            const level_offset = target.cell_offset;
            const level_total = target.cell_total;

            const level_result = system.systemStructSlice(result, level_offset, level_total);
            const level_input = system.systemStructSlice(input, level_offset, level_total);
            const level_context = system.systemStructSlice(context, level_offset, level_total);

            for (target.blocks.items(.bounds), target.blocks.items(.cell_offset), target.blocks.items(.cell_total)) |block, offset, total| {
                const block_result = system.systemStructSlice(level_result, offset, total);
                const block_input = system.systemStructSlice(level_input, offset, total);
                const block_context = system.systemStructSlice(level_context, offset, total);

                const stencil_space: StencilSpace = self.levelStencilSpace(level, block);

                var cell_indices = stencil_space.cellSpace().cells();

                while (cell_indices.next()) |cell| {
                    const offset_cell = if (full) add(stencil_space.size, [1]isize{-O} ** N) else cell;
                    const engine = EngineType(N, O, @TypeOf(oper)){
                        .inner = .{
                            .space = stencil_space,
                            .cell = offset_cell,
                        },
                        .context = block_context,
                        .operated = block_input,
                    };

                    const app: system.SystemValue(@TypeOf(oper).System) = oper.apply(engine);

                    inline for (system.systemFieldNames(@TypeOf(oper).System)) |name| {
                        stencil_space.setValue(
                            offset_cell,
                            @field(block_result, name),
                            @field(app, name),
                        );
                    }
                }
            }
        }

        // *************************
        // Projection **************
        // *************************

        pub fn project(
            self: *const Self,
            func: anytype,
            result: system.SystemSlice(@TypeOf(func).Output),
            context: system.SystemSliceConst(@TypeOf(func).Context),
        ) void {
            self.projectBase(func, result, context);

            for (0..self.active_levels) |level| {
                self.projectLevel(level, func, result, context);
            }
        }

        pub fn projectBase(
            self: *const Self,
            func: anytype,
            result: system.SystemSlice(@TypeOf(func).Output),
            context: system.SystemSliceConst(@TypeOf(func).Context),
        ) void {
            if (comptime !(operator.isMeshFunction(N, O)(@TypeOf(func)))) {
                @compileError("Func must satisfy isMeshFunction trait.");
            }

            const base_offset = 0;
            const base_total = self.base.cell_total;

            const base_result = system.systemStructSlice(result, base_offset, base_total);
            const base_context = system.systemStructSlice(context, base_offset, base_total);

            const stencil_space: StencilSpace = self.baseStencilSpace();

            var cell_indices = stencil_space.cellSpace().cells();

            while (cell_indices.next()) |cell| {
                const engine = EngineType(N, O, @TypeOf(func)){
                    .inner = .{
                        .space = stencil_space,
                        .cell = cell,
                    },
                    .context = base_context,
                };

                const val: system.SystemValue(@TypeOf(func).Output) = func.value(engine);

                inline for (comptime system.systemFieldNames(@TypeOf(func).Output)) |name| {
                    stencil_space.cellSpace().setValue(
                        cell,
                        @field(base_result, name),
                        @field(val, name),
                    );
                }
            }
        }

        pub fn projectLevel(
            self: *const Self,
            level: usize,
            func: anytype,
            result: system.SystemSlice(@TypeOf(func).Output),
            context: system.SystemSliceConst(@TypeOf(func).Context),
        ) void {
            if (comptime !(operator.isMeshFunction(N, O)(@TypeOf(func)))) {
                @compileError("Func must satisfy isMeshFunction trait.");
            }

            const target: *const Level = &self.levels.items[level];

            const level_offset = target.cell_offset;
            const level_total = target.cell_total;

            const level_result = system.systemStructSlice(result, level_offset, level_total);
            const level_context = system.systemStructSlice(context, level_offset, level_total);

            for (target.blocks.items(.cell_offset), target.blocks.items(.cell_total), 0..) |offset, total, id| {
                const block_result = system.systemStructSlice(level_result, offset, total);
                const block_context = system.systemStructSlice(level_context, offset, total);

                const stencil_space: StencilSpace = self.levelStencilSpace(level, id);

                var cell_indices = stencil_space.cellSpace().cells();

                while (cell_indices.next()) |cell| {
                    const engine = EngineType(N, O, @TypeOf(func)){
                        .inner = .{
                            .space = stencil_space,
                            .cell = cell,
                        },
                        .context = block_context,
                    };

                    const val: system.SystemValue(@TypeOf(func).Output) = func.value(engine);

                    inline for (comptime system.systemFieldNames(@TypeOf(func).Output)) |name| {
                        stencil_space.cellSpace().setValue(
                            cell,
                            @field(block_result, name),
                            @field(val, name),
                        );
                    }
                }
            }
        }

        // *************************
        // Elliptic Solve (MOVE) ***
        // *************************

        fn SolveOperator(comptime T: type, comptime B: type) type {
            _ = B;
            return struct {
                mesh: *const Self,
                oper: T,
                context: T.Context,

                fn apply(self: @This(), res: []f64, x: []const f64) void {
                    var operated: T.System = undefined;
                    @field(operated, system.systemFieldNames(T.System)[0]) = x;

                    const stencil_space = self.mesh.baseStencilSpace();
                    var cells = stencil_space.cellSpace().cells();

                    while (cells.next()) |cell| {
                        const engine = EngineType(N, O, T){
                            .inner = .{
                                .space = self.stencil_space,
                                .cell = cell,
                            },
                            .context = self.context,
                            .operated = operated,
                        };

                        stencil_space.cellSpace().setValue(cell, res, self.oper.apply(engine));
                    }
                }
            };
        }

        pub fn solveBase(
            self: *const Self,
            oper: anytype,
            result: system.SystemSlice(@TypeOf(oper).System),
            rhs: system.SystemSliceConst(@TypeOf(oper).System),
            context: system.SystemSliceConst(@TypeOf(oper).Context),
        ) void {
            if (comptime !(operator.isMeshOperator(N, O)(@TypeOf(oper)))) {
                @compileError("Oper must satisfy isMeshOperator trait.");
            }

            if (comptime system.systemFieldCount(@TypeOf(oper).System) != 1) {
                @compileError("solveBase only supports systems with 1 field currently.");
            }

            const field_name: [:0]const u8 = system.systemFieldNames(@TypeOf(oper).System)[0];

            const result_field: []f64 = @field(result, field_name);
            const rhs_field: []const f64 = @field(rhs, field_name);

            const Operator = struct {
                inner: @TypeOf(oper),
                ctx: oper.Context,
            };

            const solve_oper: Operator(@TypeOf(oper)) = .{
                .mesh = self,
                .oper = oper,
                .context = context,
            };

            var sol: solver.BiCGStabSolver(2) = solver.BiCGStabSolver(2).init(self.gpa, result_field.len, 1000, 10e-6);
            defer sol.deinit();

            sol.solve(solve_oper, result_field, rhs_field);
        }

        // *************************
        // Output ******************
        // *************************

        fn GridData(comptime System: type, comptime NVertices: usize) type {
            const field_count = system.systemFieldCount(System);
            return struct {
                positions: ArrayListUnmanaged(f64) = .{},
                vertices: ArrayListUnmanaged(usize) = .{},
                fields: [field_count]ArrayListUnmanaged(f64) = [1]ArrayListUnmanaged(f64){.{}} ** field_count,
                input: System,

                fn deinit(data: *@This(), allocator: Allocator) void {
                    data.positions.deinit(allocator);
                    data.vertices.deinit(allocator);

                    for (0..data.fields.len) |i| {
                        data.fields[i].deinit(allocator);
                    }
                }

                fn build(
                    data: *@This(),
                    allocator: Allocator,
                    stencil: StencilSpace,
                    offset: usize,
                    total: usize,
                ) !void {
                    const cell_size = stencil.size;
                    const point_size = add(cell_size, splat(1));

                    const cell_space: IndexSpace = IndexSpace.fromSize(cell_size);
                    const point_space: IndexSpace = IndexSpace.fromSize(point_size);

                    const point_offset: usize = data.positions.items.len;

                    try data.positions.ensureUnusedCapacity(allocator, N * point_space.total());
                    try data.vertices.ensureUnusedCapacity(allocator, NVertices * cell_space.total());

                    // Fill positions and vertices

                    var points = point_space.cartesianIndices();

                    while (points.next()) |point| {
                        const pos = stencil.vertexPosition(toSigned(point));
                        for (0..N) |i| {
                            data.positions.appendAssumeCapacity(pos[i]);
                        }
                    }

                    if (N == 1) {
                        var cells = cell_space.cartesianIndices();

                        while (cells.next()) |cell| {
                            const v1: usize = point_space.linearFromCartesian(cell);
                            const v2: usize = point_space.linearFromCartesian(add(cell, splat(1)));

                            data.vertices.appendAssumeCapacity(point_offset + v1);
                            data.vertices.appendAssumeCapacity(point_offset + v2);
                        }
                    } else if (N == 2) {
                        var cells = cell_space.cartesianIndices();

                        while (cells.next()) |cell| {
                            const v1: usize = point_space.linearFromCartesian(cell);
                            const v2: usize = point_space.linearFromCartesian(add(cell, [2]usize{ 0, 1 }));
                            const v3: usize = point_space.linearFromCartesian(add(cell, [2]usize{ 1, 1 }));
                            const v4: usize = point_space.linearFromCartesian(add(cell, [2]usize{ 1, 0 }));

                            data.vertices.appendAssumeCapacity(point_offset + v1);
                            data.vertices.appendAssumeCapacity(point_offset + v2);
                            data.vertices.appendAssumeCapacity(point_offset + v3);
                            data.vertices.appendAssumeCapacity(point_offset + v4);
                        }
                    } else if (N == 3) {
                        var cells = cell_space.cartesianIndices();

                        while (cells.next()) |cell| {
                            const v1: usize = point_space.linearFromCartesian(cell);
                            const v2: usize = point_space.linearFromCartesian(add(cell, [3]usize{ 0, 1, 0 }));
                            const v3: usize = point_space.linearFromCartesian(add(cell, [3]usize{ 1, 1, 0 }));
                            const v4: usize = point_space.linearFromCartesian(add(cell, [3]usize{ 1, 0, 0 }));
                            const v5: usize = point_space.linearFromCartesian(add(cell, [3]usize{ 0, 0, 1 }));
                            const v6: usize = point_space.linearFromCartesian(add(cell, [3]usize{ 0, 1, 3 }));
                            const v7: usize = point_space.linearFromCartesian(add(cell, [3]usize{ 1, 1, 3 }));
                            const v8: usize = point_space.linearFromCartesian(add(cell, [3]usize{ 1, 0, 3 }));

                            data.vertices.appendAssumeCapacity(point_offset + v1);
                            data.vertices.appendAssumeCapacity(point_offset + v2);
                            data.vertices.appendAssumeCapacity(point_offset + v3);
                            data.vertices.appendAssumeCapacity(point_offset + v4);
                            data.vertices.appendAssumeCapacity(point_offset + v5);
                            data.vertices.appendAssumeCapacity(point_offset + v6);
                            data.vertices.appendAssumeCapacity(point_offset + v7);
                            data.vertices.appendAssumeCapacity(point_offset + v8);
                        }
                    }

                    for (0..field_count) |id| {
                        try data.fields[id].ensureUnusedCapacity(allocator, cell_space.total());
                    }

                    const block_field = system.systemStructSlice(data.input, offset, total);

                    var cells = cell_space.cartesianIndices();
                    while (cells.next()) |cell| {
                        inline for (comptime system.systemFieldNames(System), 0..) |name, id| {
                            data.fields[id].appendAssumeCapacity(stencil.value(toSigned(cell), @field(block_field, name)));
                        }
                    }
                }
            };
        }

        pub fn writeVtk(self: *const Self, sys: anytype, out_stream: anytype) !void {
            if (comptime !(system.isSystemSliceConst(@TypeOf(sys)) or system.isSystemSlice(@TypeOf(sys)))) {
                @compileError("Sys must satisfy isSystemSliceConst trait.");
            }

            const vtkio = @import("../vtkio.zig");
            const VtuMeshOutput = vtkio.VtuMeshOutput;
            const VtkCellType = vtkio.VtkCellType;

            // Global Constants
            const cell_type: VtkCellType = switch (N) {
                1 => .line,
                2 => .quad,
                3 => .hexa,
                else => @compileError("Vtk Output not supported for N > 3"),
            };

            // Build data
            var data = GridData(@TypeOf(sys), cell_type.nvertices()){ .input = sys };
            defer data.deinit(self.gpa);

            // Build base
            try data.build(
                self.gpa,
                self.baseStencilSpace(),
                0,
                self.base.cell_total,
            );

            for (0..self.active_levels) |level| {
                const target: *const Level = &self.levels.items[level];
                const level_offset = target.cell_offset;
                const block_offsets = target.blocks.items(.cell_offset);
                const block_totals = target.blocks.items(.cell_total);

                for (block_offsets, block_totals, 0..) |offset, total, id| {
                    const stencil: StencilSpace = self.levelStencilSpace(level, id);

                    try data.build(
                        self.gpa,
                        stencil,
                        level_offset + offset,
                        total,
                    );
                }
            }

            var grid: VtuMeshOutput = try VtuMeshOutput.init(self.gpa, .{
                .points = data.positions.items,
                .vertices = data.vertices.items,
                .cell_type = cell_type,
            });
            defer grid.deinit();

            for (system.systemFieldNames(@TypeOf(sys)), 0..) |name, id| {
                try grid.addCellField(name, data.fields[id].items, 1);
            }

            try grid.write(out_stream);
        }

        // *************************
        // Regridding **************
        // *************************

        pub const RegridConfig = struct {
            max_levels: usize,
            block_max_tiles: usize,
            block_efficiency: f64,
            patch_max_tiles: usize,
            patch_efficiency: f64,
        };

        pub fn regrid(self: *Self, tags: []bool, config: RegridConfig) !void {
            assert(config.max_levels >= self.active_levels);

            // 1. Find total number of levels and preallocate dest.
            // **********************************************************
            const total_levels = self.computeTotalLevels(tags, config);
            try self.resizeActiveLevels(total_levels);

            // 2. Recursively generate levels on new mesh.
            // *******************************************

            // Build scratch allocator
            var arena: ArenaAllocator = ArenaAllocator.init(self.gpa);
            defer arena.deinit();

            var scratch: Allocator = arena.allocator();

            // Bounds for base level.
            const bbounds: IndexBox = .{
                .origin = [1]usize{0} ** N,
                .size = self.base.index_size,
            };

            // Slices at top of scope ensures we don't reference a temporary.
            const bbounds_slice: []const IndexBox = &[_]IndexBox{bbounds};
            const boffsets_slice: []const usize = &[_]usize{0};

            // Stores a set of clusters to consider when repartitioning l. This consists of all patches on l+1.
            var clusters: ArrayListUnmanaged(IndexBox) = .{};
            defer clusters.deinit(self.gpa);

            // A map from index in clusters to index of patch on l+1.
            var cluster_index_map: ArrayListUnmanaged(usize) = .{};
            defer cluster_index_map.deinit(self.gpa);

            // Allows mapping from coarse patch to clusters.
            var cluster_offsets: ArrayListUnmanaged(usize) = .{};
            defer cluster_offsets.deinit(self.gpa);

            // At the end of iterating level l, this contains offsets from coarse patches on l-1 to new patches on l.
            var coarse_children: ArrayListUnmanaged(usize) = .{};
            defer coarse_children.deinit(self.gpa);

            // Loop through levels from highest to lowest
            for (0..total_levels) |reverse_level_id| {
                const level_id: usize = total_levels - 1 - reverse_level_id;
                // Get a mutable reference to the target level.
                const target: *Level = &self.levels.items[level_id];

                // Check if there exists a level higher than the current one.
                const refined_exists: bool = level_id < total_levels - 1;
                // Check if we are over base level.
                const coarse_exists: bool = level_id > 0;

                // At this moment in time
                // - coarse is old
                // - target is old
                // - refined has been fully updated

                // To assemble clusters per patch we iterate children of coarse, then children of target

                clusters.shrinkRetainingCapacity(0);
                cluster_index_map.shrinkRetainingCapacity(0);
                cluster_offsets.shrinkRetainingCapacity(0);

                try cluster_offsets.append(self.gpa, 0);

                if (refined_exists and coarse_exists) {
                    const refined: *const Level = &self.levels.items[level_id + 1];
                    const coarse: *const Level = &self.levels.items[level_id - 1];

                    for (0..coarse.patches.len) |cpid| {
                        const cpbounds: IndexBox = coarse.patches.items(.bounds)[cpid];

                        for (coarse.childrenSlice(cpid)) |tpid| {
                            const start = coarse_children.items[tpid];
                            const end = coarse_children.items[tpid + 1];

                            for (start..end) |child| {
                                var patch: IndexBox = refined.patches.items(.bounds)[child];

                                try patch.coarsen();
                                try patch.coarsen();

                                try clusters.append(self.gpa, patch.relativeTo(cpbounds));
                                try cluster_index_map.append(self.gpa, child);
                            }
                        }

                        try cluster_offsets.append(self.gpa, clusters.items.len);
                    }
                } else if (refined_exists) {
                    const refined: *const Level = &self.levels.items[level_id + 1];

                    // Every patch in target is a child of base block.
                    for (0..target.patches.len) |tpid| {
                        for (target.childrenSlice(tpid)) |child| {
                            var patch: IndexBox = refined.patches.items(.bounds)[child];

                            try patch.coarsen();
                            try patch.coarsen();

                            try clusters.append(self.gpa, patch);
                            try cluster_index_map.append(self.gpa, child);
                        }
                    }

                    try cluster_offsets.append(self.gpa, clusters.items.len);
                } else {
                    try cluster_offsets.append(self.gpa, clusters.items.len);
                }

                // Now shrinkRetainingCapacitywe can clear the target arrays
                // target.transfer_blocks.shrinkRetainingCapacity(0);
                // target.transfer_patches.shrinkRetainingCapacity(0);

                // if (coarse_exists) {
                //     const coarse: *const Level = &self.levels.items[level_id - 1];

                //     for (0..coarse.patches.len) |cpid| {
                //         const upbounds: IndexBox = coarse.patches.items(.bounds)[cpid];

                //         try target.transfer_patches.append(self.gpa, TransferPatch{
                //             .bounds = upbounds,
                //             .block_offset = 0,
                //             .block_total = 0,
                //             .patch_offset = 0,
                //             .patch_total = 0,
                //         });
                //     }

                //     for (target.blocks.items(.bounds), target.blocks.items(.patch)) |bounds, patch| {
                //         const cpid: usize = coarse.parents.items[patch];
                //         target.transfer_patches.items(.block_total)[cpid] += 1;

                //         try target.transfer_blocks.append(self.gpa, TransferBlock{
                //             .bounds = bounds,
                //             .patch = cpid,
                //         });
                //     }
                // } else {
                //     try target.transfer_patches.append(self.gpa, TransferPatch{
                //         .bounds = bbounds,
                //         .block_offset = 0,
                //         .block_total = 0,
                //         .patch_offset = 0,
                //         .patch_total = 0,
                //     });

                //     for (target.blocks.items(.bounds)) |bounds| {
                //         target.transfer_patches.items(.block_total)[0] += 1;

                //         try target.transfer_blocks.append(self.gpa, TransferBlock{
                //             .bounds = bounds,
                //             .patch = 0,
                //         });
                //     }
                // }

                try target.setTotalChildren(self.gpa, clusters.items.len);
                target.clearRetainingCapacity();

                // At this moment in time
                // - coarse is old
                // - target is cleared
                // - refined has been fully updated

                // Variables that depend on coarse existing
                var ctags: []bool = undefined;
                var clen: usize = undefined;
                var cbounds: []const IndexBox = undefined;
                var coffsets: []const usize = undefined;

                if (coarse_exists) {
                    // Get underlying data
                    const coarse: *const Level = &self.levels.items[level_id - 1];

                    ctags = coarse.levelTileSlice(tags);
                    clen = coarse.patches.len;
                    cbounds = coarse.patches.items(.bounds);
                    coffsets = coarse.patches.items(.tile_offset);
                } else {
                    // Otherwise use base data
                    ctags = self.baseTileSlice(tags);
                    clen = 1;
                    cbounds = bbounds_slice;
                    coffsets = boffsets_slice;
                }

                // 3.3 Generate new patches.
                // *************************

                // Start filling coarse children
                coarse_children.shrinkRetainingCapacity(0);

                try coarse_children.append(self.gpa, 0);

                for (0..clen) |cpid| {
                    // Reset arena for new "frame"
                    defer _ = arena.reset(.retain_capacity);

                    // Make aliases for patch variables
                    const cpbounds: IndexBox = cbounds[cpid];
                    const cpoffset: usize = coffsets[cpid];
                    const cpspace: IndexSpace = IndexSpace.fromBox(cpbounds);
                    const cptags: []bool = ctags[cpoffset..(cpoffset + cpspace.total())];

                    // As well as clusters in this patch
                    const upclusters: []const IndexBox = clusters.items[cluster_offsets.items[cpid]..cluster_offsets.items[cpid + 1]];
                    const upcluster_index_map: []const usize = cluster_index_map.items[cluster_offsets.items[cpid]..cluster_offsets.items[cpid + 1]];

                    // Preprocess tags to include all elements from clusters (and one tile buffer region around cluster)
                    preprocessTagsOnPatch(cptags, cpspace, upclusters);

                    // Run partitioning algorithm on coarse patch to determine blocks.
                    var cppartitioner = try PartitionSpace.init(scratch, cpbounds.size, upclusters);
                    defer cppartitioner.deinit();

                    try cppartitioner.build(cptags, config.patch_max_tiles, config.patch_efficiency);

                    // Iterate computed patches
                    for (cppartitioner.partitions()) |patch| {
                        // Build a space from the patch size.
                        const pspace: IndexSpace = IndexSpace.fromBox(patch.bounds);

                        // Allocate sufficient space to hold children of this patch
                        var pchildren: []usize = try scratch.alloc(usize, patch.children_total);
                        defer scratch.free(pchildren);

                        // Fill with computed children of patch (using index map to find global index into refined children)
                        for (patch.children_offset..(patch.children_offset + patch.children_total), pchildren) |child_id, *child| {
                            child.* = upcluster_index_map[child_id];
                        }

                        // Build window into tags for this patch
                        var ptags: []bool = try scratch.alloc(bool, pspace.total());
                        defer scratch.free(ptags);
                        // Set ptags using window of uptags
                        cpspace.window(patch.bounds, bool, ptags, cptags);

                        // Run patitioning algorithm on patch to determine blocks
                        var ppartitioner = try PartitionSpace.init(scratch, pspace.size, &[_]IndexBox{});
                        defer ppartitioner.deinit();

                        try ppartitioner.build(ptags, config.block_max_tiles, config.block_efficiency);

                        // Offset blocks to be in global space
                        var pblocks: []IndexBox = try scratch.alloc(IndexBox, ppartitioner.partitions().len);
                        scratch.free(pblocks);

                        // Compute global bounds of the patch
                        const pbounds: IndexBox = .{
                            .origin = add(patch.bounds.origin, cpbounds.origin),
                            .size = patch.bounds.size,
                        };

                        // Iterate computed blocks and offset to find global bounds of each block
                        for (ppartitioner.partitions(), pblocks) |block, *pblock| {
                            pblock.* = block.bounds;

                            for (0..N) |axis| {
                                pblock.origin[axis] += pbounds.origin[axis];
                            }
                        }

                        // Add patch to level
                        try target.addPatch(self.gpa, pbounds, pblocks, pchildren);
                    }

                    try coarse_children.append(self.gpa, target.patchTotal());

                    // target.transfer_patches.items(.patch_total)[cpid] += partition_space.parts.len;
                }

                // if (coarse_exists) {
                //     const coarse: *Level = &self.levels.items[level_id - 1];

                //     try coarse.children.resize(self.gpa, target.patches.len);
                //     try coarse.parents.resize(self.gpa, target.patches.len);

                //     for (0..target.patches.len) |i| {
                //         coarse.children.items[i] = i;
                //     }

                //     for (0..clen) |cpid| {
                //         const start = coarse_children.items[cpid];
                //         const end = coarse_children.items[cpid + 1];

                //         coarse.patches.items(.children_offset)[cpid] = start;
                //         coarse.patches.items(.children_total)[cpid] = end - start;

                //         @memset(coarse.parents.items[start..end], cpid);
                //     }
                // }

                target.refine();

                // At this moment in time
                // - coarse is old
                // - target has been fully updated
                // - refined has been fully updated
            }

            // 4. Recompute level offsets and totals.
            // **************************************

            self.computeOffsets();
        }

        fn preprocessTagsOnPatch(tags: []bool, space: IndexSpace, clusters: []const IndexBox) void {
            for (clusters) |upcluster| {
                var cluster: IndexBox = upcluster;

                for (0..N) |i| {
                    if (cluster.origin[i] > 0) {
                        cluster.origin[i] -= 1;
                        cluster.size[i] += 1;
                    }

                    if (cluster.origin[i] + cluster.size[i] < space.size[i]) {
                        cluster.size[i] += 1;
                    }
                }

                space.fillSubspace(cluster, bool, tags, true);
            }
        }

        fn computeOffsets(self: *Self) void {
            var tile_offset: usize = self.base.tile_total;
            var cell_offset: usize = self.base.cell_total;

            for (self.levels.items) |*level| {
                level.computeOffsets(self.config.tile_width);

                level.tile_offset = tile_offset;
                level.cell_offset = cell_offset;

                tile_offset += level.tile_total;
                cell_offset += level.cell_total;
            }

            self.cell_total = cell_offset;
            self.tile_total = tile_offset;
        }

        fn computeTotalLevels(self: *const Self, tags: []const bool, config: RegridConfig) usize {
            // Clamp to max levels
            if (self.active_levels == config.max_levels) {
                return config.max_levels;
            }

            // Check if any on the highest level is tagged
            if (self.active_levels > 0) {
                for (tags[self.levels.getLast().tile_offset..]) |tag| {
                    if (tag) {
                        return self.active_levels + 1;
                    }
                }

                return self.active_levels;
            } else {
                for (tags) |tag| {
                    if (tag) {
                        return self.active_levels + 1;
                    }
                }

                return self.active_levels;
            }
        }

        fn resizeActiveLevels(self: *Self, total: usize) !void {
            while (total > self.levels.items.len) {
                if (self.levels.items.len == 0) {
                    const size: [N]usize = scaled(self.base.index_size, 2);

                    try self.levels.append(self.gpa, Level.init(size));
                } else {
                    const size: [N]usize = scaled(self.levels.getLast().index_size, 2);

                    try self.levels.append(self.gpa, Level.init(size));
                }
            }

            self.active_levels = total;
        }
    };
}

test "mesh regridding" {
    // const expect = std.testing.expect;
    // const expectEqualSlices = std.testing.expectEqualSlices;

    const allocator = std.testing.allocator;

    const Mesh2 = Mesh(2, 0);

    const config: Mesh2.Config = .{
        .physical_bounds = .{
            .origin = [_]f64{ 0.0, 0.0 },
            .size = [_]f64{ 1.0, 1.0 },
        },
        .index_size = [_]usize{ 10, 10 },
        .tile_width = 16,
    };

    var mesh: Mesh2 = Mesh2.init(allocator, config);
    defer mesh.deinit();

    var tags: []bool = try allocator.alloc(bool, mesh.tileTotal());
    defer allocator.free(tags);

    // Tag all
    @memset(tags, true);

    try mesh.regrid(tags, .{
        .max_levels = 1,
        .block_max_tiles = 80,
        .block_efficiency = 0.7,
        .patch_max_tiles = 80,
        .patch_efficiency = 0.1,
    });
}

test {
    _ = system;
}
