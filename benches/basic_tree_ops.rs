use std::time::Duration;

use aeon_tk::geometry::{CellId, HyperBox, Region, Tree};
use criterion::{Criterion, criterion_group, criterion_main};
use rand::{Rng as _, SeedableRng};

fn generate_tree() -> Tree<2> {
    let mut tree = Tree::<2>::new(HyperBox::UNIT);
    let mut rng = rand::rngs::StdRng::seed_from_u64(31415);
    for axis in 0..2 {
        tree.set_periodic(axis, rng.random());
    }
    tree.refine(&[true]);
    tree.refine(&[true, true, true, true]);

    for _ in 0..5 {
        let mut flags: Vec<bool> = std::iter::repeat_with(|| rng.random())
            .take(tree.num_leaves())
            .collect();
        tree.balance_refine_flags(&mut flags);
        tree.refine(&flags);
    }

    tree
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("basic_tree_ops");

    // Increase warm-up to 10 seconds and measurement to 20 seconds
    group.warm_up_time(Duration::from_secs(4));
    group.measurement_time(Duration::from_secs(20));

    group.bench_function("neighbors", |b| {
        b.iter_batched(
            generate_tree,
            |tree| {
                let mut accum = 0;
                for cell in 0..tree.num_cells() {
                    for region in Region::<2>::enumerate() {
                        accum += tree
                            .neighbor_region(CellId(cell), region)
                            .unwrap_or(CellId(0))
                            .0
                    }
                }
                accum
            },
            criterion::BatchSize::LargeInput,
        )
    });
    group.bench_function("neighbors_alt", |b| {
        b.iter_batched(
            generate_tree,
            |tree| {
                let mut accum = 0;
                for cell in 0..tree.num_cells() {
                    for region in Region::<2>::enumerate() {
                        accum += tree
                            .neighbor_region_alt(CellId(cell), region)
                            .unwrap_or(CellId(0))
                            .0
                    }
                }
                accum
            },
            criterion::BatchSize::LargeInput,
        )
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
