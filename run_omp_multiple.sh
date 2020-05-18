#!/bin/bash
#SBATCH --job-name=OpenMP_Run_multiple
#SBATCH --partition=intel_32
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200000
#SBATCH --cpus-per-task=40
#SBATCH --exclusive
#SBATCH --array=0-11
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.schenke@hzdr.de
#SBATCH -o log.om.slurm-%A_%a.out
#SBATCH -e err.om.slurm-%A_%a.out

set -x

# load modules
export alpaka_DIR=/home/schenk24/workspace/alpaka/
module load git gcc cmake boost python

cd build_omp
python ../run_multiple_detectors.py $SLURM_ARRAY_TASK_ID
