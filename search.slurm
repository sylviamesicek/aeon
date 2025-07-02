#! /bin/bash
#SBATCH --time=120:00:0
#SBATCH --account=belz-np
#SBATCH --partition=belz-shared-np
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --job-name=gausssearch
#SBATCH -o ./output/gausssearch.out
#SBATCH -e ./output/gausssearch.err

module load rust/1.87.0
cargo build --release
cargo run --release --package asphere -- search config/sphgauss1