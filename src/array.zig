const std = @import("std");

pub fn Array(comptime N: usize, comptime T: type) type {
    return struct {
        pub fn splat(v: T) [N]T {
            return [1]T{v} ** N;
        }

        pub fn add(v: [N]T, u: anytype) [N]T {
            var res: [N]T = v;

            for (0..N) |i| {
                res[i] += u[i];
            }

            return res;
        }

        pub fn sub(v: [N]T, u: anytype) [N]T {
            var res: [N]T = v;

            for (0..N) |i| {
                res[i] -= u[i];
            }

            return res;
        }

        pub fn scaled(v: [N]T, u: anytype) [N]T {
            var res: [N]T = v;

            for (0..N) |i| {
                res[i] += u;
            }

            return res;
        }
    };
}
