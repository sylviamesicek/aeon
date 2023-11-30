//! This module defines the concept of a system, ie a coupled set of fields defined over some physical/numerical space.
//! A system can be described with an enum specifying the names of the composite fields.
//!
//! ```
//! const System = enum {
//!     field1,
//!     field2,
//!     // ...
//! };
//! ```
//!
//! This module provides traits for identifying such types, and building new types represents values of these systems at
//! individual cells, slices of the systems in a SoA storage, ect.

const std = @import("std");

pub fn System(comptime Tag: type, comptime Payload: type) type {
    if (comptime !isSystemTag(Tag)) {
        @compileError("Tag must satisfy isSystemTag trait.");
    }

    return std.enums.EnumFieldStruct(Tag, Payload, null);
}

pub fn isSystemTag(comptime T: type) bool {
    switch (@typeInfo(T)) {
        .Enum => |info| {
            if (info.is_exhaustive) {
                return true;
            }
            return false;
        },
        else => return false,
    }
}

pub fn SystemSlice(comptime Tag: type) type {
    return System(Tag, []f64);
}

pub fn SystemSliceConst(comptime Tag: type) type {
    return System(Tag, []const f64);
}

test "system trait" {
    const expect = std.testing.expect;

    const Sys1 = struct {
        scalar: []const f64,
        other_scalar: []const f64,
        random: f64,
    };

    const Sys2 = enum {
        a,
        b,
        c,
    };

    try expect(!isSystemTag(Sys1));
    try expect(isSystemTag(Sys2));
}
