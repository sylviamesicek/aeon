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
// const boundary = @import("boundary.zig");
const engine = @import("engine.zig");
const integration = @import("integration.zig");
const nodes = @import("nodes.zig");
const system = @import("system.zig");
const traits = @import("traits.zig");

// pub const BoundaryKind = boundary.BoundaryKind;
// pub const BoundaryEngine = boundary.BoundaryEngine;
// pub const Robin = boundary.Robin;
// pub const isBoundary = boundary.isBoundary;

pub const Engine = engine.Engine;

pub const ForwardEulerIntegrator = integration.ForwardEulerIntegrator;
pub const RungeKutta4Integrator = integration.RungeKutta4Integrator;
pub const isOrdinaryDiffEq = integration.isOrdinaryDiffEq;

pub const NodeSpace = nodes.NodeSpace;
pub const Stencil = nodes.NodeOperator;

pub const System = system.System;
pub const SystemConst = system.SystemConst;
pub const isSystemTag = system.isSystemTag;

pub const isOperator = traits.isOperator;
pub const checkOperator = traits.checkOperator;
pub const IdentityOperator = traits.IdentityOperator;

pub const isFunction = traits.isFunction;
pub const checkFunction = traits.checkFunction;
pub const ZeroFunction = traits.ZeroFunction;
pub const ConstantFunction = traits.ConstantField;

pub const isAnalyticField = traits.isAnalyticField;
pub const ZeroField = traits.ZeroField;
pub const ConstantField = traits.ConstantField;

pub const isBoundary = traits.isBoundary;
pub const checkBoundary = traits.checkBoundary;
pub const NuemannBoundary = traits.NuemannBoundary;
pub const OddBoundary = traits.OddBoundary;
pub const EvenBoundary = traits.EvenBoundary;

pub const isBoundarySet = traits.isBoundarySet;
pub const checkBoundarySet = traits.checkBoundarySet;

test {
    _ = engine;
    _ = integration;
    _ = nodes;
    _ = system;
    _ = traits;
}
