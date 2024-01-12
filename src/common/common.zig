//! This module handles common aspects of all mesh routines and representations in FD codes. This provides functions
//! to apply physical boundary conditions to nodes, applying tensor products of stencils to node vectors
//! (through the `NodeSpace` type), and an API for applying operators to node vectors. Individual meshes
//! must provide functions for transfering between various kinds of node vector.

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const assert = std.debug.assert;
const panic = std.debug.panic;

const basis = @import("../basis/basis.zig");
const geometry = @import("../geometry/geometry.zig");
const lac = @import("../lac/lac.zig");

// Submodules
const boundary = @import("boundary.zig");
const engine = @import("engine.zig");
const integration = @import("integration.zig");
const nodes = @import("nodes.zig");
const system = @import("system.zig");

pub const BoundaryKind = boundary.BoundaryKind;
pub const BoundaryEngine = boundary.BoundaryEngine;
pub const Robin = boundary.Robin;
pub const isBoundary = boundary.isBoundary;

pub const Engine = engine.Engine;
pub const isOperator = engine.isOperator;
pub const isProjection = engine.isProjection;

pub const ForwardEulerIntegrator = integration.ForwardEulerIntegrator;
pub const RungeKutta4Integrator = integration.RungeKutta4Integrator;
pub const isOrdinaryDiffEq = integration.isOrdinaryDiffEq;

pub const NodeSpace = nodes.NodeSpace;

pub const System = system.System;
pub const SystemConst = system.SystemConst;
pub const isSystemTag = system.isSystemTag;

test {
    _ = boundary;
    _ = engine;
    _ = nodes;
    _ = system;
}
