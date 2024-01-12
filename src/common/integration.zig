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

    if (comptime !(hasFn("preprocess")(T) and @TypeOf(T.preprocess) == fn (T, system.System(T.Tag)) T.Error!void)) {
        return false;
    }

    if (comptime !(hasFn("derivative")(T) and @TypeOf(T.derivative) == fn (T, system.System(T.Tag), system.SystemConst(T.Tag), f64) T.Error!void)) {
        return false;
    }

    return true;
}

pub fn ForwardEulerIntegrator(comptime Tag: type) type {
    return struct {
        allocator: Allocator,
        sys: System,
        time: f64,

        const System = system.System(Tag);
        const SystemConst = system.SystemConst(Tag);

        pub fn init(allocator: Allocator, nnodes: usize) !@This() {
            var sys = try System.init(allocator, nnodes);
            errdefer sys.deinit(allocator);

            return .{
                .allocator = allocator,
                .sys = sys,
                .time = 0.0,
            };
        }

        pub fn deinit(self: *@This()) void {
            self.sys.deinit(self.allocator);
        }

        pub fn step(self: *@This(), allocator: Allocator, deriv: anytype, h: f64) !void {
            if (comptime !(isOrdinaryDiffEq(@TypeOf(deriv))) and @TypeOf(deriv).Tag == Tag) {
                @compileError("Derivative type must satisfy isOrdinaryDiffEq trait.");
            }

            const scratch = try System.init(allocator, self.sys.len);
            defer scratch.deinit(allocator);

            // Calculate derivative
            try deriv.preprocess(self.sys);
            try deriv.derivative(scratch, self.sys.toConst(), self.time);
            // Step
            inline for (comptime std.enums.values(Tag)) |field| {
                for (0..self.sys.len) |idx| {
                    self.sys.field(field)[idx] += h * scratch.field(field)[idx];
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
        time: f64,

        const System = system.System(Tag);
        const SystemConst = system.SystemConst(Tag);

        pub fn init(allocator: Allocator, nnodes: usize) !@This() {
            const sys = try System.init(allocator, nnodes);
            errdefer sys.deinit(allocator);

            return .{
                .allocator = allocator,
                .sys = sys,
                .time = 0.0,
            };
        }

        pub fn deinit(self: *@This()) void {
            self.sys.deinit(self.allocator);
        }

        pub fn step(self: *@This(), allocator: Allocator, deriv: anytype, h: f64) !void {
            if (comptime !(isOrdinaryDiffEq(@TypeOf(deriv))) and @TypeOf(deriv).Tag == Tag) {
                @compileError("Derivative type must satisfy isOrdinaryDiffEq trait.");
            }

            // Allocate scratch vectors

            const scratch = try System.init(allocator, self.sys.len);
            defer scratch.deinit(allocator);

            const k1 = try System.init(allocator, self.sys.len);
            defer k1.deinit(allocator);

            const k2 = try System.init(allocator, self.sys.len);
            defer k2.deinit(allocator);

            const k3 = try System.init(allocator, self.sys.len);
            defer k3.deinit(allocator);

            const k4 = try System.init(allocator, self.sys.len);
            defer k4.deinit(allocator);

            // Calculate k1
            try deriv.preprocess(self.sys);
            try deriv.derivative(k1, self.sys.toConst(), self.time);
            // Calculate k2
            inline for (comptime std.enums.values(Tag)) |field| {
                for (0..self.sys.len) |idx| {
                    scratch.field(field)[idx] = self.sys.field(field)[idx] + h / 2.0 * k1.field(field)[idx];
                }
            }

            try deriv.preprocess(scratch);
            try deriv.derivative(k2, scratch.toConst(), self.time + h / 2.0);

            // Calculate k3
            inline for (comptime std.enums.values(Tag)) |field| {
                for (0..self.sys.len) |idx| {
                    scratch.field(field)[idx] = self.sys.field(field)[idx] + h / 2.0 * k2.field(field)[idx];
                }
            }

            try deriv.preprocess(scratch);
            try deriv.derivative(k3, scratch, self.time + h / 2.0);

            // Calculate k4
            inline for (comptime std.enums.values(Tag)) |field| {
                for (0..self.sys.len) |idx| {
                    scratch.field(field)[idx] = self.sys.field(field)[idx] + h * k3.field(field)[idx];
                }
            }

            try deriv.preprocess(scratch);
            try deriv.derivative(k4, scratch, self.time + h);

            // Update sys
            inline for (comptime std.enums.values(Tag)) |field| {
                for (0..self.sys.len) |idx| {
                    self.sys.field(field)[idx] += h / 6.0 * (k1.field(field)[idx] + 2.0 * k2.field(field)[idx] + 2.0 * k3.field(field)[idx] + k4.field(field)[idx]);
                }
            }

            self.time += h;
        }
    };
}
