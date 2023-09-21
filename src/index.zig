const std = @import("std");

pub fn Index(comptime N: usize) type {
    return struct {
        pub fn splat(v: usize) [N]usize {
            return [1]usize{v} ** N;
        }

        pub fn add(v: [N]usize, u: [N]usize) [N]usize {
            var res: [N]usize = v;

            for (0..N) |i| {
                res[i] += u[i];
            }

            return res;
        }

        pub fn sub(v: [N]usize, u: [N]usize) [N]usize {
            var res: [N]usize = v;

            for (0..N) |i| {
                res[i] -= u[i];
            }

            return res;
        }

        pub fn scaled(v: [N]usize, u: usize) [N]usize {
            var res: [N]usize = v;

            for (0..N) |i| {
                res[i] *= u;
            }

            return res;
        }

        pub fn refined(self: [N]usize) [N]usize {
            var result: [N]usize = undefined;
            for (0..N) |i| {
                result[i] = self[i] * 2;
            }
            return result;
        }

        pub fn coarsened(self: [N]usize) [N]usize {
            var result: [N]usize = undefined;
            for (0..N) |i| {
                result[i] = self[i] / 2;
            }
            return result;
        }

        pub fn toSigned(self: [N]usize) [N]isize {
            var result: [N]isize = undefined;
            for (0..N) |i| {
                result[i] = @intCast(self[i]);
            }
            return result;
        }
    };
}
