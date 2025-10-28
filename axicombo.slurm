#! /bin/bash
#SBATCH --time=40:00:0
#SBATCH --account=belz-np
#SBATCH --partition=belz-shared-np
#SBATCH --ntasks=9
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=axicombo
#SBATCH -o ./output/axicombo.out
#SBATCH -e ./output/axicombo.err
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=sylvia.mesicek@utah.edu

module load rust/1.87.0
module load gcc mpich
cargo build --release --package aaxi --features "mpi"

mpirun -n $SLURM_NTASKS ./target/release/aaxi search config/axicombo -Dbrill=0.0