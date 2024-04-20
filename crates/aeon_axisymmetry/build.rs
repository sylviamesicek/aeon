use std::ffi::c_double;
use std::mem::size_of;

use cc::Build;

fn main() {
    if size_of::<c_double>() != 8 {
        panic!("Double data type on architecture is not f64.")
    }

    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let eqs_dir = manifest_dir.join("eqs");

    Build::new()
        .warnings(false)
        .include(&eqs_dir)
        .file(&eqs_dir.join("eqs.c"))
        .compile("axisymmetric_eqs");
}
