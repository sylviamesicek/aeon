const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const lac = @import("lac.zig");

const IdentityMap = lac.IdentityMap;
const isLinearMap = lac.isLinearMap;
const hasLinearMapCallback = lac.hasLinearMapCallback;
const isLinearSolver = lac.isLinearSolver;

/// A solver which uses the Bi-conjugate gradient method allong with `L` iterations of
/// GMRES to smooth the solution vector between steps.
pub fn BiCGStablSolver(comptime L: usize) type {
    const zdim: usize = L + 1;
    return struct {
        allocator: Allocator,
        ndofs: usize,
        max_iters: usize,
        tolerance: f64,

        // Scratch vectors
        rtld: []f64,
        bp: []f64,
        t: []f64,
        tp: []f64,
        xp: []f64,

        // Used for GMRES smoothing
        r: [zdim][]f64,
        u: [zdim][]f64,

        const Self = @This();

        pub fn init(allocator: Allocator, ndofs: usize, max_iters: usize, tolerance: f64) !Self {
            const rtld: []f64 = try allocator.alloc(f64, ndofs);
            errdefer allocator.free(rtld);

            const bp: []f64 = try allocator.alloc(f64, ndofs);
            errdefer allocator.free(bp);

            const t: []f64 = try allocator.alloc(f64, ndofs);
            errdefer allocator.free(t);

            const tp: []f64 = try allocator.alloc(f64, ndofs);
            errdefer allocator.free(tp);

            const xp: []f64 = try allocator.alloc(f64, ndofs);
            errdefer allocator.free(xp);

            var r: [zdim][]f64 = [1][]f64{&[_]f64{}} ** zdim;
            var u: [zdim][]f64 = [1][]f64{&[_]f64{}} ** zdim;

            errdefer {
                for (0..zdim) |i| {
                    allocator.free(r[i]);
                    allocator.free(u[i]);
                }
            }

            for (0..zdim) |i| {
                r[i] = try allocator.alloc(f64, ndofs);
                u[i] = try allocator.alloc(f64, ndofs);
            }

            return Self{
                .allocator = allocator,
                .ndofs = ndofs,
                .max_iters = max_iters,
                .tolerance = tolerance,

                .rtld = rtld,
                .bp = bp,
                .t = t,
                .tp = tp,
                .xp = xp,

                .r = r,
                .u = u,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.rtld);
            self.allocator.free(self.bp);
            self.allocator.free(self.t);
            self.allocator.free(self.tp);
            self.allocator.free(self.xp);

            for (0..zdim) |i| {
                self.allocator.free(self.r[i]);
                self.allocator.free(self.u[i]);
            }
        }

        pub fn solve(self: *const Self, oper: anytype, x: []f64, b: []const f64) void {
            assert(x.len == self.ndofs);
            assert(b.len == self.ndofs);

            var tau: [zdim * zdim]f64 = undefined;
            var gamma: [zdim]f64 = undefined;
            var gamma1: [zdim]f64 = undefined;
            var gamma2: [zdim]f64 = undefined;
            var sigma: [zdim]f64 = undefined;

            // Set termination tolerance
            oper.apply(self.tp, x);

            for (0..self.ndofs) |i| {
                self.r[0][i] = b[i] - self.tp[i];
            }

            // Set initial guesses
            @memcpy(self.rtld, self.r[0]);
            @memcpy(self.bp, self.r[0]);
            @memcpy(self.xp, x);
            @memset(self.u[0], 0.0);

            var nrm2: f64 = norm2(self.r[0]);
            var iter: usize = 0;

            end: {
                var finish_flag: bool = false;

                if (nrm2 == 0.0) {
                    finish_flag = true;
                }

                if (finish_flag) break :end;

                // Init
                var tol: f64 = @fabs(self.tolerance) * nrm2;
                var alpha: f64 = 0.0;
                var beta: f64 = 0.0;
                var omega: f64 = 1.0;
                var rho0: f64 = 1.0;
                var rho1: f64 = 0.0;

                while (iter < self.max_iters) : (iter += 1) {
                    // BiCG Part
                    // rho0 = -w*rho0
                    rho0 = -omega * rho0;

                    for (0..L) |j| {
                        iter += 1;

                        // rho1 = <rtld, r[j]>
                        rho1 = dot(self.rtld, self.r[j]);

                        // Test for breakdown
                        if (@fabs(rho1) == 0.0) {
                            // Precondition (Not implemented yet)
                            // @memcpy(self.t, self.x);
                            // @memset(self.t, 0.0);

                            for (x, self.xp) |*xval, xpval| {
                                xval.* += xpval;
                            }

                            finish_flag = true;
                        }

                        if (finish_flag) break :end;

                        // beta = alpha * (rho1/rho0)
                        // rho0 = rho1
                        beta = alpha * (rho1 / rho0);
                        rho0 = rho1;

                        // u[i] = r[i] - beta*u[i] from i in [0, j]
                        for (0..(j + 1)) |i| {
                            for (self.u[i], self.r[i]) |*uval, rval| {
                                uval.* = -beta * uval.* + rval;
                            }
                        }

                        // TODO Preconditioning

                        // u[j + 1] = M^-1 * A * u[j]
                        oper.apply(self.u[j + 1], self.u[j]);

                        // nu = <rtld, u[j + 1]>
                        const nu = dot(self.rtld, self.u[j + 1]);

                        // Test for breakdown
                        if (@fabs(nu) == 0.0) {
                            for (x, self.xp) |*xval, xpval| {
                                xval.* += xpval;
                            }

                            finish_flag = true;
                        }

                        // Alpha = rho1 / nu
                        alpha = rho1 / nu;

                        // x += alpha * u[0]
                        for (x, self.u[0]) |*xval, uval| {
                            xval.* += alpha * uval;
                        }

                        // r[i] = r[i] - alpha * u[i + 1] for i in [0, j]
                        for (0..(j + 1)) |i| {
                            for (self.r[i], self.u[i + 1]) |*rval, uval| {
                                rval.* += -alpha * uval;
                            }
                        }

                        nrm2 = norm2(self.r[0]);

                        // Stopping condition
                        if (nrm2 <= tol) {
                            // Precondition (Not implemented yet)
                            for (x, self.xp) |*xval, xpval| {
                                xval.* += xpval;
                            }

                            finish_flag = true;
                        }

                        if (finish_flag) break :end;

                        // r[j + 1] = M^-1 * A * r[j]
                        oper.apply(self.r[j + 1], self.r[j]);
                    }

                    // MR Part
                    var j: usize = 1;
                    while (j <= L) : (j += 1) {
                        var i: usize = 1;
                        while (i <= j - 1) : (i += 1) {
                            const nu = dot(self.r[j], self.r[i]) / sigma[i];
                            tau[i * zdim + j] = nu;

                            for (self.r[j], self.r[i]) |*rj, ri| {
                                rj.* = rj.* - nu * ri;
                            }
                        }

                        sigma[j] = dot(self.r[j], self.r[j]);
                        const nu = dot(self.r[0], self.r[j]);
                        gamma1[j] = nu / sigma[j];
                    }

                    gamma[L] = gamma1[L];
                    omega = gamma[L];

                    j = L - 1;
                    while (j >= 1) : (j -= 1) {
                        var nu: f64 = 0.0;
                        var i: usize = j + 1;
                        while (i <= L) : (i += 1) {
                            nu += tau[j * zdim + i] * gamma[i];
                        }
                        gamma[j] = gamma1[j] - nu;
                    }

                    j = 1;
                    while (j <= L - 1) : (j += 1) {
                        var nu: f64 = 0.0;
                        var i: usize = j + 1;
                        while (i <= L - 1) : (i += 1) {
                            nu += tau[j * zdim + i] * gamma[i + 1];
                        }
                        gamma2[j] = gamma[j + 1] + nu;
                    }

                    // Update
                    axpby(gamma[1], self.r[0], 1, x);
                    axpby(-gamma1[L], self.r[L], 1, self.r[0]);
                    axpby(-gamma[L], self.u[L], 1, self.u[0]);

                    j = 1;
                    while (j <= L - 1) : (j += 1) {
                        axpby(-gamma[j], self.u[j], 1, self.u[0]);
                        axpby(gamma2[j], self.r[j], 1, x);
                        axpby(-gamma1[j], self.r[j], 1, self.r[0]);
                    }

                    if (comptime hasLinearMapCallback(@TypeOf(oper))) {
                        oper.callback(iter, nrm2, x);
                    }

                    if (nrm2 < tol) {
                        // Precondition (Not implemented yet)
                        for (x, self.xp) |*xval, xpval| {
                            xval.* += xpval;
                        }

                        finish_flag = true;
                    }

                    if (finish_flag) break :end;
                }
            }

            if (iter < self.max_iters) iter += 1;

            // self.niters = iter;
            // self.res = nrm2 / ires;
        }

        fn norm2(slice: []const f64) f64 {
            return dot(slice, slice);
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
}

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

    var solver: BiCGStablSolver(2) = try BiCGStablSolver(2).init(allocator, ndofs, 1000, 10e-10);
    defer solver.deinit();

    solver.solve(IdentityMap{}, x, b);

    try expectEqualSlices(f64, b, x);
}
