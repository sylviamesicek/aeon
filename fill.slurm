#! /bin/bash
#SBATCH --time=120:00:0
#SBATCH --account=belz-np
#SBATCH --partition=belz-shared-np
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --job-name=gauss1fill
#SBATCH -o ./output/gauss1fill.out
#SBATCH -e ./output/gauss1fill.err

module load rust/1.87.0
cargo build --release
cargo run --release --package asphere -- fill config/sphgauss1-24