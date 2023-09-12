const std = @import("std");
const pow = std.math.pow;
const assert = std.debug.assert;

const lagrange = @import("lagrange.zig");
const space = @import("space.zig");

pub const StencilSpace = space.StencilSpace;
pub const InterpolationSpace = space.InterpolationSpace;
