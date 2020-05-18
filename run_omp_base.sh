#!/bin/bash
#SBATCH --job-name=OpenMP_Run_base
#SBATCH --partition=intel_32
#SBATCH --time=7:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200000
#SBATCH --cpus-per-task=40
#SBATCH --exclusive
#SBATCH --array=0-9
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.schenke@hzdr.de
#SBATCH -o log.ob.slurm-%A_%a.out
#SBATCH -e err.ob.slurm-%A_%a.out

set -x

# load modules
export alpaka_DIR=/home/schenk24/workspace/alpaka/
module load git gcc cmake boost python

cd build_omp
python ../run_base.py $SLURM_ARRAY_TASK_ID
