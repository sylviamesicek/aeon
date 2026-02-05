#! /bin/bash
#SBATCH --time=46:00:0
#SBATCH --account=belz
#SBATCH --partition=notchpeak
#SBATCH --ntasks=9
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=axi
#SBATCH -o ./output/axicombo.0.0.out
#SBATCH -e ./output/axicombo.0.0.err
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=sylvia.mesicek@utah.edu

module load rust/1.87.0
module load gcc mpich
cargo build --release --package aaxi --features "mpi"

mpirun -n $SLURM_NTASKS ./target/release/aaxi search config/axicombo -Dbrill=0.0