const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const lac = @import("lac.zig");

const IdentityMap = lac.IdentityMap;
const isLinearMap = lac.isLinearMap;
const isLinearSolver = lac.isLinearSolver;

pub const BiCGStabSolver = struct {
    allocator: Allocator,
    ndofs: usize,
    max_iters: usize,
    tolerance: f64,

    // Scratch vectors
    rg: []f64,
    rh: []f64,
    pg: []f64,
    ph: []f64,
    sg: []f64,
    sh: []f64,
    tg: []f64,
    vg: []f64,
    tp: []f64,

    const Self = @This();

    pub fn init(allocator: Allocator, ndofs: usize, max_iters: usize, tolerance: f64) !Self {
        const rg: []f64 = try allocator.alloc(f64, ndofs);
        errdefer allocator.free(rg);

        const rh: []f64 = try allocator.alloc(f64, ndofs);
        errdefer allocator.free(rh);

        const pg: []f64 = try allocator.alloc(f64, ndofs);
        errdefer allocator.free(pg);

        const ph: []f64 = try allocator.alloc(f64, ndofs);
        errdefer allocator.free(ph);

        const sg: []f64 = try allocator.alloc(f64, ndofs);
        errdefer allocator.free(sg);

        const sh: []f64 = try allocator.alloc(f64, ndofs);
        errdefer allocator.free(sh);

        const tg: []f64 = try allocator.alloc(f64, ndofs);
        errdefer allocator.free(tg);

        const vg: []f64 = try allocator.alloc(f64, ndofs);
        errdefer allocator.free(vg);

        const tp: []f64 = try allocator.alloc(f64, ndofs);
        errdefer allocator.free(tp);

        return Self{
            .allocator = allocator,
            .ndofs = ndofs,
            .max_iters = max_iters,
            .tolerance = tolerance,

            .rg = rg,
            .rh = rh,
            .pg = pg,
            .ph = ph,
            .sg = sg,
            .sh = sh,
            .tg = tg,
            .vg = vg,
            .tp = tp,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.rg);
        self.allocator.free(self.rh);
        self.allocator.free(self.pg);
        self.allocator.free(self.ph);
        self.allocator.free(self.sg);
        self.allocator.free(self.sh);
        self.allocator.free(self.tg);
        self.allocator.free(self.vg);
        self.allocator.free(self.tp);
    }

    pub fn solve(self: *const Self, oper: anytype, x: []f64, b: []const f64) void {
        assert(x.len == self.ndofs);
        assert(b.len == self.ndofs);

        // Set termination tolerance
        oper.apply(self.tp, x);

        for (0..self.ndofs) |i| {
            self.rg[i] = b[i] - self.tp[i];
        }

        @memcpy(self.rh, self.rg);
        @memset(self.sh, 0.0);
        @memset(self.ph, 0.0);

        var residual = norm2(self.rg);
        const tol = residual * @fabs(self.tolerance);

        var iter: usize = 0;
        var rho0: f64 = 0.0;
        var rho1: f64 = 0.0;
        var pra: f64 = 0.0;
        var prb: f64 = 0.0;
        var prc: f64 = 0.0;

        while (iter < self.max_iters) : (iter += 1) {
            std.debug.print("Iteration {}\n", .{iter});

            rho1 = dot(self.rg, self.rh);

            if (rho1 == 0.0) {
                break;
            }

            if (iter == 0) {
                @memcpy(self.pg, self.rg);
            } else {
                prb = (rho1 * pra) / (rho0 * prc);

                for (0..self.ndofs) |i| {
                    self.pg[i] = self.rg[i] + prb * (self.pg[i] - prc * self.vg[i]);
                }
            }

            rho0 = rho1;

            // Identity PC
            @memcpy(self.ph, self.pg);
            oper.apply(self.vg, self.ph);

            pra = rho1 / dot(self.rh, self.vg);

            for (0..self.ndofs) |i| {
                self.sg[i] = self.rg[i] - pra * self.vg[i];
            }

            if (norm2(self.sg) <= 1e-60) {
                for (0..self.ndofs) |i| {
                    x[i] = x[i] + pra * self.ph[i];
                }

                oper.apply(self.tp, x);

                for (0..self.ndofs) |i| {
                    self.rg[i] = b[i] - self.tp[i];
                }

                residual = norm2(self.rg);

                break;
            }

            @memcpy(self.sh, self.sg);
            oper.apply(self.tg, self.sh);

            prc = dot(self.tg, self.sg) / dot(self.tg, self.tg);
            for (0..self.ndofs) |i| {
                x[i] = x[i] + pra * self.ph[i] + prc * self.sh[i];
                self.rg[i] = self.sg[i] - prc * self.tg[i];
            }

            residual = norm2(self.rg);

            if (residual <= tol) break;
        }

        if (iter < self.max_iters) iter += 1;

        // self.niters = iter;
        // self.res = nrm2 / ires;
    }

    fn norm2(slice: []const f64) f64 {
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
    const expectEqualSlices = std.testing.expectEqualSlices;

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

    var solver: BiCGStabSolver = try BiCGStabSolver.init(allocator, ndofs, 1000, 1e-10);
    defer solver.deinit();

    solver.solve(IdentityMap{}, x, b);

    try expectEqualSlices(f64, b, x);
}
