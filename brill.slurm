#! /bin/bash
#SBATCH --time=120:00:0
#SBATCH --account=belz-np
#SBATCH --partition=belz-shared-np
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --job-name=brillsearch
#SBATCH -o ./output/brillsearch.out
#SBATCH -e ./output/brillsearch.err

module load rust
cargo build --release
cargo run --release --package critaxi -- "config/brillcrit.toml"