#! /bin/bash
#SBATCH --time=120:00:0
#SBATCH --account=belz-np
#SBATCH --partition=belz-shared-np
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --job-name=prolatesearch
#SBATCH -o ./output/spherecrit.out
#SBATCH -e ./output/spherecrit.err

module load rust/1.87.0
cargo build --release
cargo run --release --package asphere -- --config="config/sphgauss1-fill.toml"