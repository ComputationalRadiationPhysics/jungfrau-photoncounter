#!/bin/bash
#SBATCH --job-name=OpenMP_Run_detailed
#SBATCH --partition=defq
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200000
#SBATCH --cpus-per-task=40
#SBATCH --exclusive
#SBATCH --array=0-61
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.schenke@hzdr.de
#SBATCH -o log.od.slurm-%A_%a.out
#SBATCH -e err.od.slurm-%A_%a.out

set -x

# load modules
export alpaka_DIR=/home/schenk24/workspace/alpaka/
module load git gcc cmake boost python

cd build_omp
python ../run_detailed.py $SLURM_ARRAY_TASK_ID
