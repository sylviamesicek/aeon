use aeon_tk::prelude::*;
use criterion::{Criterion, criterion_group, criterion_main};
use rand::{Rng as _, SeedableRng};

#[derive(Clone)]
struct Custom;

impl Boundary<2> for Custom {
    fn kind(&self, _: usize, _: Face<2>) -> BoundaryKind {
        BoundaryKind::Custom
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(1027);

    // Generate Random Mesh
    let mut mesh: Mesh<2> = Mesh::new(
        HyperBox::UNIT,
        4,
        4,
        2,
        BoundaryClasses::splat(BoundaryClass::Ghost),
    );
    let num_node_per_cell = (mesh.width() + 1).pow(2);

    mesh.refine_global();
    mesh.refine_global();

    for _ in 0..5 {
        mesh.set_refine_flags_random(&mut rng);
        mesh.balance_flags();
        mesh.regrid();
    }

    let mut source = Image::new(1, mesh.num_nodes());
    rng.fill(&mut source);
    mesh.fill_boundary(Custom, source.as_mut());

    let mut base_image = Image::new(1, num_node_per_cell);

    let mut group = c.benchmark_group("basic_mesh_ops");

    // Increase warm-up to 10 seconds and measurement to 20 seconds
    // group.warm_up_time(Duration::from_secs(4));
    // group.measurement_time(Duration::from_secs(20));

    group.bench_function("load_nodes_for_cell", |b| {
        b.iter_batched(
            || CellId(rng.random_range(0..mesh.tree().num_cells())),
            |cell| {
                mesh.load_nodes_for_cell(cell, source.as_ref(), base_image.as_mut(), [true, true]);
                _ = std::hint::black_box(base_image.as_mut());
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
