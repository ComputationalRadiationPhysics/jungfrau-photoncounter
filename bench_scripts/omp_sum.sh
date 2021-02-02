#!/bin/bash
#SBATCH --job-name=SumCpu
#SBATCH --partition=defq
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200000
#SBATCH --cpus-per-task=80
#SBATCH --exclusive
#SBATCH --array=0-4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.schenke@hzdr.de
#SBATCH -o log.intel_clustercount.slurm-%A_%a.out
#SBATCH -e err.intel_clustercount.slurm-%A_%a.out

set -x

# load modules
export alpaka_DIR=/home/schenk24/workspace/alpaka/
module load git intel cmake boost python

export KMP_AFFINITY="verbose,compact"

#SUMFRAMES=("0" "2" "10" "20" "100")
SUMENABLE=("0" "1" "1" "1" "1")

cd ../build_omp
./bench s${SLURM_ARRAY_TASK_ID} 100 12.4 1 1 0 ${SUMENABLE[${SLURM_ARRAY_TASK_ID}]} ../../../data_pool/px_101016/allpede_250us_1243__B_000000.dat ../../../data_pool/px_101016/gainMaps_M022.bin ../../../data_pool/px_101016/Insu_6_tr_1_45d_250us__B_000000.dat sumframes${SLURM_ARRAY_TASK_ID}
