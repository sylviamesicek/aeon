#! /bin/bash
#SBATCH --time=120:00:0
#SBATCH --account=belz-np
#SBATCH --partition=belz-shared-np
#SBATCH --nodes=9
#SBATCH --ntasks=4
#SBATCH --mem=8G
#SBATCH --job-name=axicombo
#SBATCH -o ./output/axicombo.out
#SBATCH -e ./output/axicombo.err

module load rust/1.87.0
cargo run --release --package aaxi -- search config/axicombo -Dbrill=0.01