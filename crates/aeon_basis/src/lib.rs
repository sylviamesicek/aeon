mod boundary;
mod convolution;
mod element;
mod kernel;
mod node;

pub use boundary::{Boundary, BoundaryKind, Condition, BC};
pub use convolution::{Convolution, Dissipation, Gradient, Hessian};
pub use element::Element;
pub use kernel::{Border, CellKernel, Kernel, Kernels, Order, Value, VertexKernel};
pub use node::{
    node_from_vertex, vertex_from_node, NodeCartesianIter, NodePlaneIter, NodeSpace, NodeWindow,
};
