use std::ffi::c_double;
use std::mem::size_of;

use cc::Build;

fn main() {
    if size_of::<c_double>() != 8 {
        panic!("Double data type on architecture is not f64.")
    }

    println!("cargo::rerun-if-changed=symbolicc/symc.c");
    println!("cargo::rerun-if-changed=symbolicc/hyperbolic.h");
    println!("cargo::rerun-if-changed=symbolicc/hyperbolic_regular.h");
    println!("cargo::rerun-if-changed=symbolicc/geometric.h");

    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let eqs_dir = manifest_dir.join("symbolicc");

    Build::new()
        .warnings(false)
        .opt_level(3)
        .include(&eqs_dir)
        .file(&eqs_dir.join("symc.c"))
        .compile("symbolicc");
}
