#!/bin/bash --login
#SBATCH --job-name=reverseGOL-mpi
#SBATCH --ntasks=50
#SBATCH --mem=100gb
#SBATCH --time=00:32:00
#SBATCH --output=slurm-%j.out

module load OpenMPI

cd $SLURM_SUBMIT_DIR

make clean
make revGOL-mpi

# Run the MPI program and save output to mpi_best.txt
srun ./revGOL-mpi cmse2.txt > mpi_best.txt 
