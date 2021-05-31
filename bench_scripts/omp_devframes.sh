#!/bin/bash
#SBATCH --job-name=DevFramesCpu
#SBATCH --partition=defq
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200000
#SBATCH --cpus-per-task=80
#SBATCH --exclusive
#SBATCH --array=0-2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.schenke@hzdr.de
#SBATCH -o log.intel_clustercount.slurm-%A_%a.out
#SBATCH -e err.intel_clustercount.slurm-%A_%a.out

set -x

# load modules
export alpaka_DIR=/home/schenk24/workspace/alpaka/
module load git intel cmake boost python

export KMP_AFFINITY="verbose,compact"
export OMP_NUM_THREADS=40

cd ../build_omp
./bench d${SLURM_ARRAY_TASK_ID} 100 12.4 1 1 0 0 ../../../data_pool/px_101016/allpede_250us_1243__B_000000.dat ../../../data_pool/px_101016/gainMaps_M022.bin ../../../data_pool/px_101016/Insu_6_tr_1_45d_250us__B_000000.dat devframes${SLURM_ARRAY_TASK_ID}
