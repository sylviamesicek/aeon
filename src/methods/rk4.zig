const std = @import("std");
const Allocator = std.mem.Allocator;

const system = @import("system.zig");
const methods = @import("methods.zig");

pub fn RungeKutta4Integrator(comptime Tag: type) type {
    return struct {
        allocator: Allocator,
        sys: System,
        scratch: System,
        k1: System,
        k2: System,
        k3: System,
        k4: System,
        time: f64,

        const System = system.System(Tag);
        const SystemConst = system.SystemConst(Tag);

        pub fn init(allocator: Allocator, ndofs: usize) !@This() {
            var sys = try System.init(allocator, ndofs);
            errdefer sys.deinit(allocator);

            var scratch = try System.init(allocator, ndofs);
            errdefer scratch.deinit(allocator);

            var k1 = try System.init(allocator, ndofs);
            errdefer k1.deinit(allocator);

            var k2 = try System.init(allocator, ndofs);
            errdefer k2.deinit(allocator);

            var k3 = try System.init(allocator, ndofs);
            errdefer k3.deinit(allocator);

            var k4 = try System.init(allocator, ndofs);
            errdefer k4.deinit(allocator);

            return .{
                .allocator = allocator,
                .sys = sys,
                .scratch = scratch,
                .k1 = k1,
                .k2 = k2,
                .k3 = k3,
                .k4 = k4,
                .time = 0.0,
            };
        }

        pub fn deinit(self: *@This()) void {
            self.scratch.deinit(self.allocator);
            self.k1.deinit(self.allocator);
            self.k2.deinit(self.allocator);
            self.k3.deinit(self.allocator);
            self.k4.deinit(self.allocator);
        }

        pub fn step(self: *@This(), deriv: anytype, h: f64) void {
            if (comptime !(methods.isTemporalDerivative(@TypeOf(deriv))) and @TypeOf(deriv).Tag == Tag) {
                @compileError("Derivative type must satisfy isTemporalDerivative trait.");
            }

            // Calculate k1
            deriv.derivative(self.k1, self.sys.toConst(), self.time);
            // Calculate k2
            inline for (comptime std.enums.values(System)) |field| {
                for (0..self.sys.len) |idx| {
                    self.scratch.field(field)[idx] = self.sys.field(field)[idx] + h / 2.0 * self.k1.field(field)[idx];
                }
            }

            deriv.derivative(self.k2, self.scratch.toConst(), self.time + h / 2.0);
            // Calculate k3
            inline for (comptime std.enums.values(System)) |field| {
                for (0..self.sys.len) |idx| {
                    self.scratch.field(field)[idx] = self.sys.field(field)[idx] + h / 2.0 * self.k2.field(field)[idx];
                }
            }

            deriv.derivative(self.k3, self.scratch.toConst(), self.time + h / 2.0);
            // Calculate k4
            inline for (comptime std.enums.values(System)) |field| {
                for (0..self.sys.len) |idx| {
                    self.scratch.field(field)[idx] = self.sys.field(field)[idx] + h * self.k3.field(field)[idx];
                }
            }

            deriv.derivative(self.k4, self.scratch.toConst(), self.time + h);

            // Update sys
            inline for (comptime std.enums.values(System)) |field| {
                for (0..self.sys.len) |idx| {
                    self.sys.field(field)[idx] += h / 6.0 * (self.k1.field(field)[idx] + 2.0 * self.k2.field(field)[idx] + 2.0 * self.k3.field(field)[idx] + self.k4.field(field)[idx]);
                }
            }

            self.time += h;
        }
    };
}
