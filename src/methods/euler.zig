const std = @import("std");
const Allocator = std.mem.Allocator;

const system = @import("system.zig");
const methods = @import("methods.zig");

pub fn ForwardEulerIntegrator(comptime Tag: type) type {
    return struct {
        allocator: Allocator,
        sys: System,
        scratch: System,
        time: f64,

        const System = system.System(Tag);
        const SystemConst = system.SystemConst(Tag);

        pub fn init(allocator: Allocator, ndofs: usize) !@This() {
            var sys = try System.init(allocator, ndofs);
            errdefer sys.deinit(allocator);

            var scratch = try System.init(allocator, ndofs);
            errdefer scratch.deinit(allocator);

            return .{
                .allocator = allocator,
                .sys = sys,
                .scratch = scratch,
                .time = 0.0,
            };
        }

        pub fn deinit(self: *@This()) void {
            self.sys.deinit(self.allocator);
            self.scratch.deinit(self.allocator);
        }

        pub fn step(self: *@This(), deriv: anytype, h: f64) void {
            if (comptime !(methods.isTemporalDerivative(@TypeOf(deriv))) and @TypeOf(deriv).Tag == Tag) {
                @compileError("Derivative type must satisfy isTemporalDerivative trait.");
            }

            // Calculate derivative
            deriv.derivative(self.scratch, self.sys, self.time);
            // Step
            inline for (comptime std.enums.values(Tag)) |field| {
                for (0..self.sys.len) |idx| {
                    self.sys.field(field)[idx] += h * self.scratch.field(field)[idx];
                }
            }

            self.time += h;
        }
    };
}
