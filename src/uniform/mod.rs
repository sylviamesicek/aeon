use crate::{
    common::Engine,
    geometry::{Face, FaceArray, HyperBox, IndexSpace, IndexWindow},
    image::{ImageMut, ImageRef},
    kernel::{Border, Boundary, BoundaryKind, Kernel},
};

// const DISS_SIX_ORDER_0_7: [f64; 8] = [4.0, -27.0, 78.0, -125.0, 120.0, -69.0, 22.0, -3.0];
// const DISS_SIX_ORDER_1_6: [f64; 8] = [3.0, -20.0, 57.0, -90.0, 85.0, -48.0, 15.0, -2.0];
// const DISS_SIX_ORDER_2_5: [f64; 8] = [2.0, -13.0, 36.0, -55.0, 50.0, -27.0, 8.0, -1.0];
// const DISS_SIX_ORDER_3_3: [f64; 7] = [1.0, -6.0, 15.0, -20.0, 15.0, -6.0, 1.0];
// const DISS_SIX_ORDER_5_2: [f64; 8] = [-1.0, 8.0, -27.0, 50.0, -55.0, 36.0, -13.0, 2.0];
// const DISS_SIX_ORDER_6_1: [f64; 8] = [-2.0, 15.0, -48.0, 85.0, -90.0, 57.0, -20.0, 3.0];

pub struct UniformGrid<const N: usize> {
    /// What physical bounds does this cover?
    bounds: HyperBox<N>,
    /// Number of nodes along each axis
    size: [usize; N],
    /// Number of ghost nodes padding each face.
    ghost: usize,
    /// Labels for each boundary in the grid.
    bidx: FaceArray<N, usize>,
}

impl<const N: usize> UniformGrid<N> {
    pub fn num_nodes(&self) -> usize {
        self.space().count()
    }

    pub fn space(&self) -> IndexSpace<N> {
        IndexSpace::new(std::array::from_fn(|axis| self.size[axis] + 2 * self.ghost))
    }

    pub fn interior(&self) -> IndexWindow<N> {
        IndexWindow {
            origin: [self.ghost; N],
            size: self.size,
        }
    }

    /// Returns the spacing along each axis of the node space.
    pub fn spacing(&self) -> [f64; N] {
        std::array::from_fn(|axis| self.bounds.size[axis] / (self.size[axis] - 1) as f64)
    }

    /// Computes the position of the given node.
    pub fn position(&self, node: [usize; N]) -> [f64; N] {
        let spacing = self.spacing();
        std::array::from_fn(|i| self.bounds.origin[i] + spacing[i] * node[i] as f64)
    }

    /// Applies strong boundary conditions to the given image.
    pub fn apply_boundary<B: Boundary<N>>(&self, boundary: B, mut image: ImageMut) {
        debug_assert!(image.len() == self.num_nodes());

        for face in Face::iterate() {
            // Retrieve index for a given face
            let index = self.bidx[face];
            // Compute window to fill for parity conditions
            let mut parity_window = self.space().window();
            parity_window.size[face.axis] = self.ghost;
            if face.side {
                parity_window.origin[face.axis] = self.ghost + self.size[face.axis];
            }
            // Compute window to fill for diritchlet conditions
            let mut dirichlet_window = self.space().window();
            dirichlet_window.size[face.axis] = 1;
            parity_window.origin[face.axis] = self.ghost;
            if face.side {
                parity_window.origin[face.axis] += self.size[face.axis] - 1;
            }

            // Loop over channels
            for channel in image.channels() {
                let output = image.channel_mut(channel);

                let kind = boundary.kind(index, channel);
                match kind {
                    BoundaryKind::AntiSymmetric | BoundaryKind::Symmetric => {
                        let sign = if kind == BoundaryKind::Symmetric {
                            1.0
                        } else {
                            -1.0
                        };

                        for node in parity_window {
                            let mut source = node;
                            if face.side {
                                let dist =
                                    node[face.axis] - (self.size[face.axis] + self.ghost - 1);
                                source[face.axis] -= 2 * dist;
                            } else {
                                let dist = self.ghost - node[face.axis];
                                source[face.axis] += 2 * dist;
                            }

                            output[self.space().linear_from_cartesian(node)] =
                                output[self.space().linear_from_cartesian(source)]
                        }
                    }
                    BoundaryKind::Dirichlet => {
                        for node in dirichlet_window {
                            let position = self.position(node);
                            let target = boundary.dirichlet(channel, position);
                            output[self.space().linear_from_cartesian(node)] = target;
                        }
                    }
                    BoundaryKind::Free => {}
                }

                if kind == BoundaryKind::AntiSymmetric {
                    for node in dirichlet_window {
                        let position = self.position(node);
                        output[self.space().linear_from_cartesian(node)] = 0.0;
                    }
                }
            }
        }
    }

    /// Computes the result of applying a kernel to a specific node of an image, assuming certain boundary conditions.
    pub fn kernel<K: Kernel, B: Boundary<N>>(
        &self,
        mut node: [usize; N],
        axis: usize,
        kernel: K,
        boundary: B,
        image: ImageRef,
        channel: usize,
    ) -> f64 {
        debug_assert!(node[axis] >= self.ghost && node[axis] < self.ghost + self.size[axis]);

        // Spacing along axis
        let spacing = self.bounds.size[axis] / (self.size[axis] - 1) as f64;

        let positive_kind = boundary.kind(self.bidx[Face::positive(axis)], channel);
        let negative_kind = boundary.kind(self.bidx[Face::negative(axis)], channel);

        if positive_kind.is_one_sided() {
            let distance = self.size[axis] - self.ghost - 1 - node[axis];

            if distance < kernel.border_width() {
                let stencil = kernel.free(Border::Positive(distance));
                node[axis] = self.ghost + self.size[axis] - stencil.len();
                return self.stencil(node, axis, stencil, image.channel(channel))
                    * kernel.scale(spacing);
            }
        }

        if negative_kind.is_one_sided() {
            let distance = node[axis] - self.ghost;

            if distance < kernel.border_width() {
                let stencil = kernel.free(Border::Negative(distance));
                node[axis] = self.ghost;
                return self.stencil(node, axis, stencil, image.channel(channel))
                    * kernel.scale(spacing);
            }
        }

        let stencil = kernel.interior();
        node[axis] -= kernel.border_width();
        self.stencil(node, axis, stencil, image.channel(channel)) * kernel.scale(spacing)
    }

    /// Applies a single stencil to a channel along an axis, extending rightwards from the provided node.
    pub fn stencil(&self, node: [usize; N], axis: usize, stencil: &[f64], channel: &[f64]) -> f64 {
        debug_assert!(node[axis] >= self.ghost && node[axis] < self.ghost + self.size[axis]);

        let mut result = 0.0;

        for (i, weight) in stencil.iter().enumerate() {
            let mut source = node;
            source[axis] = node[axis] + i;
            let linear = self.space().linear_from_cartesian(source);

            result += channel[linear] * weight;
        }

        result
    }

    /// Performs kernel operations along an axis in bulk on the source image, storing the result in dest.
    pub fn convolve<K: Kernel, B: Boundary<N>>(
        &self,
        kernel: K,
        axis: usize,
        boundary: B,
        source: ImageRef,
        mut dest: ImageMut,
    ) {
        debug_assert!(self.num_nodes() == source.len());
        debug_assert!(self.num_nodes() == dest.len());
        debug_assert!(source.num_channels() == dest.num_channels());

        // Spacing along axis
        let spacing = self.bounds.size[axis] / (self.size[axis] - 1) as f64;

        let num_channels = source.num_channels();

        for channel in 0..num_channels {
            let source = source.channel(channel);
            let dest = dest.channel_mut(channel);

            let negative_kind = boundary.kind(self.bidx[Face::negative(axis)], channel);
            let positive_kind = boundary.kind(self.bidx[Face::positive(axis)], channel);

            let negative_edge = if negative_kind.is_one_sided() {
                self.ghost + kernel.border_width()
            } else {
                self.ghost
            };
            let positive_edge = if positive_kind.is_one_sided() {
                self.ghost + self.size[axis] - kernel.border_width()
            } else {
                self.ghost + self.size[axis]
            };

            let mut window = self.space().window();
            window.size[axis] = 1;

            for mut node in window {
                for i in self.ghost..negative_edge {
                    let distance = i - self.ghost;
                    let stencil = kernel.free(Border::Negative(distance));
                    node[axis] = self.ghost;

                    let mut output = node;
                    output[axis] = i;

                    dest[self.space().linear_from_cartesian(output)] =
                        self.stencil(node, axis, stencil, source) * kernel.scale(spacing);
                }

                for i in negative_edge..positive_edge {
                    let stencil = kernel.interior();
                    node[axis] = i - kernel.border_width();

                    let mut output = node;
                    output[axis] = i;

                    dest[self.space().linear_from_cartesian(output)] =
                        self.stencil(node, axis, stencil, source) * kernel.scale(spacing);
                }

                for i in positive_edge..self.ghost + self.size[axis] {
                    let distance = self.ghost + self.size[axis] - 1 - i;
                    let stencil = kernel.free(Border::Positive(distance));
                    node[axis] = self.ghost + self.size[axis] - stencil.len();

                    let mut output = node;
                    output[axis] = i;

                    dest[self.space().linear_from_cartesian(output)] =
                        self.stencil(node, axis, stencil, source) * kernel.scale(spacing);
                }
            }
        }
    }
}

pub struct UniformEngine<'a, const N: usize, const ORDER: usize, B> {
    grid: &'a UniformGrid<N>,
    boundary: &'a B,
    source: ImageRef<'a>,
    gradient: ImageRef<'a>,
    hessian: ImageRef<'a>,
    node: [usize; N],
}
