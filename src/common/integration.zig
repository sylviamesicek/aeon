const std = @import("std");
const Allocator = std.mem.Allocator;

const system = @import("system.zig");

/// A trait for determining if a type consitutes an ordinary differential equation.
pub fn isOrdinaryDiffEq(comptime T: type) bool {
    const hasFn = std.meta.hasFn;

    if (comptime !(@hasDecl(T, "Tag") and @TypeOf(T.Tag) == type and system.isSystemTag(T.Tag))) {
        return false;
    }

    if (comptime !(@hasDecl(T, "Error") and @TypeOf(T.Error) == type)) {
        return false;
    }

    if (comptime !(hasFn(T, "preprocess") and @TypeOf(T.preprocess) == fn (T, system.System(T.Tag)) T.Error!void)) {
        return false;
    }

    if (comptime !(hasFn(T, "derivative") and @TypeOf(T.derivative) == fn (T, system.System(T.Tag), system.SystemConst(T.Tag), f64) T.Error!void)) {
        return false;
    }

    return true;
}

pub fn ForwardEulerIntegrator(comptime Tag: type) type {
    return struct {
        allocator: Allocator,
        sys: System,
        scr: System,
        time: f64,

        const System = system.System(Tag);
        const SystemConst = system.SystemConst(Tag);

        pub fn init(allocator: Allocator, nnodes: usize) !@This() {
            var sys = try System.init(allocator, nnodes);
            errdefer sys.deinit(allocator);

            var scr = try System.init(allocator, nnodes);
            errdefer scr.deinit(allocator);

            return .{
                .allocator = allocator,
                .sys = sys,
                .scr = scr,
                .time = 0.0,
            };
        }

        pub fn deinit(self: *@This()) void {
            self.sys.deinit(self.allocator);
            self.scr.deinit(self.allocator);
        }

        pub fn step(self: *@This(), deriv: anytype, h: f64) !void {
            if (comptime !(isOrdinaryDiffEq(@TypeOf(deriv))) and @TypeOf(deriv).Tag == Tag) {
                @compileError("Derivative type must satisfy isOrdinaryDiffEq trait.");
            }

            // Calculate derivative
            try deriv.preprocess(self.sys);
            try deriv.derivative(self.scr, self.sys.toConst(), self.time);
            // Step
            inline for (comptime std.enums.values(Tag)) |field| {
                for (0..self.sys.len) |idx| {
                    self.sys.field(field)[idx] += h * self.scr.field(field)[idx];
                }
            }

            self.time += h;
        }
    };
}

pub fn RungeKutta4Integrator(comptime Tag: type) type {
    return struct {
        allocator: Allocator,
        sys: System,
        scr: System,
        k1: System,
        k2: System,
        k3: System,
        k4: System,
        time: f64,

        const System = system.System(Tag);
        const SystemConst = system.SystemConst(Tag);

        pub fn init(allocator: Allocator, nnodes: usize) !@This() {
            const sys = try System.init(allocator, nnodes);
            errdefer sys.deinit(allocator);

            const scr = try System.init(allocator, nnodes);
            errdefer scr.deinit(allocator);

            const k1 = try System.init(allocator, nnodes);
            errdefer k1.deinit(allocator);

            const k2 = try System.init(allocator, nnodes);
            errdefer k2.deinit(allocator);

            const k3 = try System.init(allocator, nnodes);
            errdefer k3.deinit(allocator);

            const k4 = try System.init(allocator, nnodes);
            errdefer k4.deinit(allocator);

            return .{
                .allocator = allocator,
                .sys = sys,
                .scr = scr,
                .k1 = k1,
                .k2 = k2,
                .k3 = k3,
                .k4 = k4,
                .time = 0.0,
            };
        }

        pub fn deinit(self: *@This()) void {
            self.sys.deinit(self.allocator);
            self.scr.deinit(self.allocator);
            self.k1.deinit(self.allocator);
            self.k2.deinit(self.allocator);
            self.k3.deinit(self.allocator);
            self.k4.deinit(self.allocator);
        }

        pub fn step(self: *@This(), deriv: anytype, h: f64) !void {
            if (comptime !(isOrdinaryDiffEq(@TypeOf(deriv))) and @TypeOf(deriv).Tag == Tag) {
                @compileError("Derivative type must satisfy isOrdinaryDiffEq trait.");
            }

            // Calculate k1
            try deriv.preprocess(self.sys);
            try deriv.derivative(self.k1, self.sys.toConst(), self.time);

            // Calculate k2
            inline for (comptime std.enums.values(Tag)) |field| {
                for (0..self.sys.len) |idx| {
                    self.scr.field(field)[idx] = self.sys.field(field)[idx] + h / 2.0 * self.k1.field(field)[idx];
                }
            }

            try deriv.preprocess(self.scr);
            try deriv.derivative(self.k2, self.scr.toConst(), self.time + h / 2.0);

            // Calculate k3
            inline for (comptime std.enums.values(Tag)) |field| {
                for (0..self.sys.len) |idx| {
                    self.scr.field(field)[idx] = self.sys.field(field)[idx] + h / 2.0 * self.k2.field(field)[idx];
                }
            }

            try deriv.preprocess(self.scr);
            try deriv.derivative(self.k3, self.scr.toConst(), self.time + h / 2.0);

            // Calculate k4
            inline for (comptime std.enums.values(Tag)) |field| {
                for (0..self.sys.len) |idx| {
                    self.scr.field(field)[idx] = self.sys.field(field)[idx] + h * self.k3.field(field)[idx];
                }
            }

            try deriv.preprocess(self.scr);
            try deriv.derivative(self.k4, self.scr.toConst(), self.time + h);

            // Update sys
            inline for (comptime std.enums.values(Tag)) |field| {
                for (0..self.sys.len) |idx| {
                    self.sys.field(field)[idx] += h / 6.0 * (self.k1.field(field)[idx] + 2.0 * self.k2.field(field)[idx] + 2.0 * self.k3.field(field)[idx] + self.k4.field(field)[idx]);
                }
            }

            self.time += h;
        }
    };
}
