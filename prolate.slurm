#! /bin/bash
#SBATCH --time=120:00:0
#SBATCH --account=belz-np
#SBATCH --partition=belz-shared-np
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=prolatesearch
#SBATCH -o ./output/prolatesearch.out
#SBATCH -e ./output/prolatesearch.err

module load rust
cargo build --release
cargo run --release --package critaxi -- "config/prolatecrit.toml"