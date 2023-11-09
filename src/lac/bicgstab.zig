const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const lac = @import("lac.zig");

const IdentityMap = lac.IdentityMap;
const isLinearMap = lac.isLinearMap;
const hasLinearMapCallback = lac.hasLinearMapCallback;
const isLinearSolver = lac.isLinearSolver;

pub const BiCGStabSolver = struct {
    max_iters: usize,
    tolerance: f64,

    // Alias
    const Self = @This();

    // Constructs a new bicgstab solver
    pub fn new(max_iters: usize, tolerance: f64) Self {
        return .{
            .max_iters = max_iters,
            .tolerance = tolerance,
        };
    }

    pub const Error = error{OutOfMemory};

    pub fn solve(self: *const Self, allocator: Allocator, oper: anytype, x: []f64, b: []const f64) Error!void {
        assert(x.len == b.len);

        // Compute tolerance
        const tol = @fabs(self.tolerance) * norm(b);

        if (norm(b) <= 1e-60) {
            @memset(x, 0.0);
            return;
        }

        // Allocate scratch vectors
        const ndofs = x.len;

        const rg: []f64 = try allocator.alloc(f64, ndofs);
        defer allocator.free(rg);

        const rh: []f64 = try allocator.alloc(f64, ndofs);
        defer allocator.free(rh);

        const pg: []f64 = try allocator.alloc(f64, ndofs);
        defer allocator.free(pg);

        const ph: []f64 = try allocator.alloc(f64, ndofs);
        defer allocator.free(ph);

        const sg: []f64 = try allocator.alloc(f64, ndofs);
        defer allocator.free(sg);

        const sh: []f64 = try allocator.alloc(f64, ndofs);
        defer allocator.free(sh);

        const tg: []f64 = try allocator.alloc(f64, ndofs);
        defer allocator.free(tg);

        const vg: []f64 = try allocator.alloc(f64, ndofs);
        defer allocator.free(vg);

        const tp: []f64 = try allocator.alloc(f64, ndofs);
        defer allocator.free(tp);

        // Set termination tolerance
        oper.apply(tp, x);

        for (0..ndofs) |i| {
            rg[i] = b[i] - tp[i];
        }

        @memcpy(rh, rg);
        @memset(sh, 0.0);
        @memset(ph, 0.0);

        var residual = norm(rg);

        var iter: usize = 0;
        var rho0: f64 = 0.0;
        var rho1: f64 = 0.0;
        var pra: f64 = 0.0;
        var prb: f64 = 0.0;
        var prc: f64 = 0.0;

        while (iter < self.max_iters) : (iter += 1) {
            rho1 = dot(rg, rh);

            if (rho1 == 0.0) {
                break;
            }

            if (iter == 0) {
                @memcpy(pg, rg);
            } else {
                prb = (rho1 * pra) / (rho0 * prc);

                for (0..ndofs) |i| {
                    pg[i] = rg[i] + prb * (pg[i] - prc * vg[i]);
                }
            }

            rho0 = rho1;

            // Identity PC
            @memcpy(ph, pg);
            oper.apply(vg, ph);

            pra = rho1 / dot(rh, vg);

            for (0..ndofs) |i| {
                sg[i] = rg[i] - pra * vg[i];
            }

            if (norm(sg) <= 1e-60) {
                for (0..ndofs) |i| {
                    x[i] = x[i] + pra * ph[i];
                }

                oper.apply(tp, x);

                for (0..ndofs) |i| {
                    rg[i] = b[i] - tp[i];
                }

                residual = norm(rg);

                break;
            }

            @memcpy(sh, sg);
            oper.apply(tg, sh);

            prc = dot(tg, sg) / dot(tg, tg);
            for (0..ndofs) |i| {
                x[i] = x[i] + pra * ph[i] + prc * sh[i];
                rg[i] = sg[i] - prc * tg[i];
            }

            residual = norm(rg);

            if (comptime hasLinearMapCallback(@TypeOf(oper))) {
                oper.callback(iter, residual, x);
            }

            if (residual <= tol) break;
        }

        if (iter < self.max_iters) iter += 1;
    }

    fn norm(slice: []const f64) f64 {
        return @sqrt(dot(slice, slice));
    }

    fn dot(u: []const f64, v: []const f64) f64 {
        var result: f64 = 0.0;
        for (u, v) |a, b| {
            result += a * b;
        }
        return result;
    }

    fn axpby(a: f64, x: []const f64, b: f64, y: []f64) void {
        for (x, y) |xv, *yv| {
            yv.* = xv * a + yv.* * b;
        }
    }
};

test "BiCGStab convergence" {
    const expect = std.testing.expect;
    const expectEqualSlices = std.testing.expectEqualSlices;

    try expect(isLinearSolver(BiCGStabSolver));

    const allocator = std.testing.allocator;
    const ndofs = 100;

    const x = try allocator.alloc(f64, ndofs);
    defer allocator.free(x);

    const b = try allocator.alloc(f64, ndofs);
    defer allocator.free(b);

    for (0..ndofs) |i| {
        x[i] = 0.0;
        b[i] = @floatFromInt(i);
    }

    const solver = BiCGStabSolver.new(1000, 1e-10);

    try solver.solve(allocator, IdentityMap{}, x, b);

    try expectEqualSlices(f64, b, x);
}
