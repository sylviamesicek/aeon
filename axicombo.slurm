#! /bin/bash
#SBATCH --time=30:00:0
#SBATCH --account=owner-guest
#SBATCH --partition=notchpeak-guest
#SBATCH --nodes=9
#SBATCH --ntasks=9
#SBATCH --mem=8G
#SBATCH --job-name=axicombo
#SBATCH -o ./output/axicombo.out
#SBATCH -e ./output/axicombo.err
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=sylvia.mesicek@utah.edu

module load rust/1.87.0
module load gcc mpich
cargo build --release --package aaxi --features "mpi"

mpirun -n $SLURM_NTASKS ./target/release/aaxi search config/axicombo -Dbrill=0.0