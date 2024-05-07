#!/bin/bash -l
#SBATCH --job-name="plankiformer-test"
#SBATCH --account="em09"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=benno.kaech@eawag.ch
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread
#SBATCH --output=slurm/slurm_%j.out
#SBATCH --error=slurm/slurm_%j.err
export OMP_NUM_THREADS=12 #$SLURM_CPUS_PER_TASK
cd /users/bkch/Plankiformer_OOD
module load daint-gpu cray-python

source ./predict.sh Mar2019