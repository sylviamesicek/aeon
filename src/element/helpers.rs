use faer::diag::generic::Diag;
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::matmul::matmul;
use faer::linalg::svd::{
    ComputeSvdVectors, SvdError, pseudoinverse_from_svd, pseudoinverse_from_svd_scratch, svd,
    svd_scratch,
};
use faer::{Accum, Mat, MatMut, MatRef, Par};
use reborrow::{Reborrow, ReborrowMut};

/// A faer workspace
pub struct Workspace {
    req: StackReq,
    buffer: MemBuffer,
}

impl Workspace {
    pub fn empty() -> Self {
        Self {
            req: StackReq::empty(),
            buffer: MemBuffer::new(StackReq::empty()),
        }
    }

    pub fn stack(&mut self, req: StackReq) -> &mut MemStack {
        if self.req.or(req) != self.req {
            self.req = req;
            self.buffer = MemBuffer::new(req);
        }

        MemStack::new(&mut self.buffer)
    }
}

impl Clone for Workspace {
    fn clone(&self) -> Self {
        Self {
            req: self.req,
            buffer: MemBuffer::new(self.req),
        }
    }
}

#[derive(Clone)]
pub struct LeastSquares {
    /// Cache for workspace memory used when computing svd.
    workspace: Workspace,
    s: Vec<f64>,
    u: Mat<f64>,
    v: Mat<f64>,
    pinv: Mat<f64>,
}

impl Default for LeastSquares {
    fn default() -> Self {
        Self {
            workspace: Workspace::empty(),
            s: Vec::default(),
            u: Mat::zeros(0, 0),
            v: Mat::zeros(0, 0),
            pinv: Mat::zeros(0, 0),
        }
    }
}

impl LeastSquares {
    fn compute_psuedo_inverse(&mut self, m: MatRef<f64>) -> Result<(), SvdError> {
        // We can only compute the psuedo inverse for overdetermined systems.
        assert!(m.nrows() >= m.ncols());

        let compute = ComputeSvdVectors::Full;
        let par = Par::Seq;

        let nrows = m.nrows();
        let ncols = m.ncols();

        // Compute memory requirements
        let svd_reqs = svd_scratch::<f64>(
            nrows,
            ncols,
            compute,
            compute,
            par,
            faer::prelude::default(),
        );
        let pinv_regs = pseudoinverse_from_svd_scratch::<f64>(nrows, ncols, par);
        let stack = self
            .workspace
            .stack(StackReq::any_of(&[svd_reqs, pinv_regs]));

        let size = nrows.min(ncols);
        self.s.resize(size, 0.0);
        self.u.resize_with(nrows, nrows, |_, _| 0.0);
        self.v.resize_with(ncols, ncols, |_, _| 0.0);

        svd(
            m,
            Diag::from_slice_mut(&mut self.s),
            Some(self.u.rb_mut()),
            Some(self.v.rb_mut()),
            par,
            stack,
            faer::prelude::default(),
        )?;

        self.pinv.resize_with(ncols, nrows, |_, _| 0.0);
        pseudoinverse_from_svd(
            self.pinv.rb_mut(),
            Diag::from_slice(&mut self.s),
            self.u.rb(),
            self.v.rb(),
            par,
            stack,
        );

        Ok(())
    }

    pub fn overdetermined(
        &mut self,
        m: MatRef<f64>,
        a: MatMut<f64>,
        b: MatRef<f64>,
    ) -> Result<(), SvdError> {
        assert!(a.nrows() == m.ncols() && a.ncols() == b.ncols() && b.nrows() == m.nrows());
        assert!(m.nrows() >= m.ncols());

        self.compute_psuedo_inverse(m)?;
        matmul(a, Accum::Replace, self.pinv.rb(), b, 1.0, Par::Seq);

        Ok(())
    }

    pub fn underdetermined(
        &mut self,
        m: MatRef<f64>,
        a: MatMut<f64>,
        b: MatRef<f64>,
    ) -> Result<(), SvdError> {
        assert!(a.nrows() == m.ncols() && a.ncols() == b.ncols() && b.nrows() == m.nrows());
        assert!(m.nrows() <= m.ncols());

        self.compute_psuedo_inverse(m.transpose())?;
        matmul(a, Accum::Replace, self.pinv.transpose(), b, 1.0, Par::Seq);

        Ok(())
    }

    pub fn least_squares(
        &mut self,
        m: MatRef<f64>,
        a: MatMut<f64>,
        b: MatRef<f64>,
    ) -> Result<(), SvdError> {
        if m.nrows() >= m.ncols() {
            self.overdetermined(m, a, b)
        } else {
            self.underdetermined(m, a, b)
        }
    }
}
