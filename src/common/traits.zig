//! Namespace for various traits used throughout this library.

const std = @import("std");
const hasFn = std.meta.hasFn;

const geometry = @import("../geometry/geometry.zig");

const boundary = @import("boundary.zig");
const engine_ = @import("engine.zig");

const BoundaryKind = boundary.BoundaryKind;
const BoundaryPolarity = boundary.BoundaryPolarity;
const Engine = engine_.Engine;

// ********************************
// Operators and Functions

/// A trait for defining operators.
pub fn isOperator(comptime N: usize, comptime M: usize, comptime T: type) bool {
    if (comptime !(@hasDecl(T, "order") and @TypeOf(T.order) == usize)) {
        return false;
    }

    const O = T.order;

    if (O > M) {
        return false;
    }

    if (comptime !(hasFn(T, "apply") and @TypeOf(T.apply) == fn (T, Engine(N, M, O), []const f64) f64)) {
        return false;
    }

    if (comptime !(hasFn(T, "applyDiag") and @TypeOf(T.applyDiag) == fn (T, Engine(N, M, O)) f64)) {
        return false;
    }

    return true;
}

/// Checks whether a type satisfies the isOperator at comptime, optionally asserting that
/// it is defined for a certain dimension, number of ghost points, or order.
pub fn checkOperator(comptime N: usize, comptime M: usize, comptime T: type) void {
    if (comptime !isOperator(N, M, T)) {
        const msg = std.fmt.comptimePrint("Type {} must satisfy isOperator trait.", .{T});
        @compileError(&(msg.*));
    }
}

pub fn IdentityOperator(comptime N: usize, comptime M: usize, comptime O: usize) type {
    return struct {
        pub const order = O;

        pub fn apply(_: @This(), engine: Engine(N, M, O), field: []const f64) f64 {
            return engine.value(field);
        }

        pub fn applyDiag(_: @This(), engine: Engine(N, M, O)) f64 {
            return engine.valueDiag();
        }
    };
}

pub fn isFunction(comptime N: usize, comptime M: usize, comptime T: type) bool {
    comptime {
        if (!(@hasDecl(T, "order") and @TypeOf(T.order) == usize)) {
            return false;
        }

        const O = T.order;

        if (O > M) {
            return false;
        }

        if (!(hasFn(T, "eval") and @TypeOf(T.eval) == fn (T, Engine(N, M, O)) f64)) {
            return false;
        }

        return true;
    }
}

pub fn checkFunction(comptime N: usize, comptime M: usize, comptime T: type) void {
    if (comptime !isFunction(N, M, T)) {
        const msg = std.fmt.comptimePrint("Type {} must satisfy isFunction trait.", .{T});
        @compileError(&(msg.*));
    }
}

pub fn ZeroFunction(comptime N: usize, comptime M: usize) type {
    return struct {
        pub const order: usize = 0;

        pub fn eval(_: @This(), _: Engine(N, M, 0)) f64 {
            return 0.0;
        }
    };
}

pub fn ConstantFunction(comptime N: usize, comptime M: usize) type {
    return struct {
        value: f64,

        pub const order: usize = 0;

        pub fn eval(self: @This(), _: Engine(N, M, 0)) f64 {
            return self.value;
        }
    };
}

pub fn checkOrder(comptime T: type, comptime order: usize) void {
    if (comptime order != T.order) {
        @compileError("Type order mismatch.");
    }
}

// *********************************************
// Analytic Fields

pub fn isAnalyticField(comptime N: usize, comptime T: type) bool {
    if (comptime !(hasFn(T, "eval") and @TypeOf(T.eval) == fn (T, [N]f64) f64)) {
        return false;
    }

    return true;
}

pub fn ZeroField(comptime N: usize) type {
    return struct {
        pub fn eval(_: @This(), _: [N]f64) f64 {
            return 0.0;
        }
    };
}

pub fn ConstantField(comptime N: usize) type {
    return struct {
        value: f64,

        pub fn eval(self: @This(), _: [N]f64) f64 {
            return self.value;
        }
    };
}

pub fn checkAnalyticField(comptime N: usize, comptime T: type) void {
    comptime {
        if (!isAnalyticField(N, T)) {
            @compileError("Type must satisfy isAnalyticField trait.");
        }
    }
}

// *********************************************
// Boundaries

/// A trait for defining boundaries
pub fn isBoundary(comptime N: usize, comptime T: type) bool {
    const fieldInfo = std.meta.fieldInfo;

    if (!(@hasDecl(T, "kind") and @TypeOf(T.kind) == BoundaryKind)) {
        return false;
    }

    if (!(@hasDecl(T, "priority") and @TypeOf(T.priority) == usize)) {
        return false;
    }

    if (T.kind == .robin) {
        if (!(@hasField(T, "robin_value") and @hasField(T, "robin_rhs"))) {
            return false;
        }

        const RobinValueType: type = fieldInfo(T, .robin_value).type;
        const RobinRhsType: type = fieldInfo(T, .robin_rhs).type;

        if (!(RobinValueType == []const f64 or isAnalyticField(N, RobinValueType))) {
            return false;
        }

        if (!(RobinRhsType == []const f64 or isAnalyticField(N, RobinRhsType))) {
            return false;
        }
    } else if (T.kind == .symmetric) {
        if (!(@hasField(T, "polarity"))) {
            return false;
        }

        if (fieldInfo(T, .polarity).type != BoundaryPolarity) {
            return false;
        }
    }

    return true;
}

pub fn checkBoundary(comptime N: usize, comptime T: type) void {
    comptime {
        if (!isBoundary(N, T)) {
            const msg = std.fmt.comptimePrint("Type {} must satisfy isBoundary trait.", .{T});
            @compileError(&(msg.*));
        }
    }
}

/// Homogenous Nuemann boundary Condition
pub fn NuemannBoundary(comptime N: usize) type {
    return struct {
        comptime robin_value: ZeroField(N) = .{},
        comptime robin_rhs: ZeroField(N) = .{},

        pub const kind: BoundaryKind = .robin;
        pub const priority: usize = 0;
    };
}

/// Antisymmetric boundary
pub const OddBoundary = struct {
    comptime polarity: BoundaryPolarity = .odd,

    pub const kind: BoundaryKind = .symmetric;
    pub const priority: usize = 1;
};

/// Symmetric Boundary
pub const EvenBoundary = struct {
    comptime polarity: BoundaryPolarity = .even,

    pub const kind: BoundaryKind = .symmetric;
    pub const priority: usize = 1;
};

/// A trait for defining boundary sets, i.e. a collection of boundaries along with a mapping from faces
/// to boundaries
pub fn isBoundarySet(comptime N: usize, comptime T: type) bool {
    const FaceIndex = geometry.FaceIndex(N);

    if (!(@hasDecl(T, "card") and @TypeOf(T.card) == usize)) {
        return false;
    }

    if (!(@hasDecl(T, "boundaryIdFromFace") and @TypeOf(T.boundaryIdFromFace) == fn (FaceIndex) usize)) {
        return false;
    }

    inline for (0..T.card) |i| {
        const type_name = std.fmt.comptimePrint("BoundaryType{}", .{i});
        const func_name = std.fmt.comptimePrint("boundary{}", .{i});

        if (!(@hasDecl(T, &(type_name.*)) and @TypeOf(@field(T, &(type_name.*))) == type)) {
            return false;
        }

        const BoundaryType = @field(T, &(type_name.*));

        if (!(@TypeOf(BoundaryType) == type and isBoundary(N, BoundaryType))) {
            return false;
        }

        if (!(hasFn(T, &(func_name.*)) and @TypeOf(@field(T, &(func_name.*))) == fn (T) BoundaryType)) {
            return false;
        }
    }

    return true;
}

/// Asserts at comptime that the given type satisfies `isBoundarySet(N)` and has compatible boundaries.
pub fn checkBoundarySet(comptime N: usize, comptime T: type) void {
    comptime {
        if (!isBoundarySet(N, T)) {
            @compileError("T must satisfy isBoundarySet trait.");
        }

        const max_priority = maxBoundaryPriority(T);

        var priorities: [max_priority + 1]?BoundaryKind = [1]?BoundaryKind{null} ** (max_priority + 1);

        for (0..T.card) |id| {
            const BoundaryType = BoundaryTypeFromId(T, id);

            if (priorities[BoundaryType.priority]) |kind| {
                if (kind != BoundaryType.kind) {
                    @compileError("Boundary Set has incompatible boundaries.");
                }
            } else {
                priorities[BoundaryType.priority] = BoundaryType.kind;
            }
        }
    }
}

/// Retrieves the type of the `id`th component of the boundary set.
pub fn BoundaryTypeFromId(comptime Set: type, comptime id: usize) type {
    if (comptime id >= Set.card) {
        @compileError("Id must be less than Boundary Set cardinality.");
    }

    const type_name = std.fmt.comptimePrint("BoundaryType{}", .{id});

    return @field(Set, type_name);
}

/// Retrieves the `id`th component of the boundary set.
pub fn boundaryFromId(set: anytype, comptime id: usize) BoundaryTypeFromId(@TypeOf(set), id) {
    const Set = @TypeOf(set);

    if (comptime id >= Set.card) {
        @compileError("Id must be less than Boundary Set cardinality.");
    }

    const func_name = std.fmt.comptimePrint("boundary{}", .{id});

    return @field(Set, func_name)(set);
}

/// Finds the maximum priority of any boundary in the set.
pub fn maxBoundaryPriority(comptime Set: type) usize {
    var result: usize = 0;

    for (0..Set.card) |id| {
        const BoundaryType: type = BoundaryTypeFromId(Set, id);

        result = @max(result, BoundaryType.priority);
    }

    return result;
}

test "traits" {
    checkOperator(2, 2, IdentityOperator(2, 2, 2));
    checkOrder(IdentityOperator(2, 2, 2), 2);
    checkFunction(2, 2, ZeroFunction(2, 2));
    checkFunction(2, 2, ConstantFunction(2, 2));
    checkBoundary(2, NuemannBoundary(2));
    checkBoundary(2, OddBoundary);
    checkBoundary(2, EvenBoundary);
}
