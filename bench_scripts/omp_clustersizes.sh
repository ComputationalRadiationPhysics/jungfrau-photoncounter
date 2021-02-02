#!/bin/bash
#SBATCH --job-name=ClusterSizeCpu
#SBATCH --partition=defq
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200000
#SBATCH --cpus-per-task=80
#SBATCH --exclusive
#SBATCH --array=0-3
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.schenke@hzdr.de
#SBATCH -o log.intel_clustercount.slurm-%A_%a.out
#SBATCH -e err.intel_clustercount.slurm-%A_%a.out

set -x

# load modules
export alpaka_DIR=/home/schenk24/workspace/alpaka/
module load git intel cmake boost python

export KMP_AFFINITY="verbose,compact"

CLUSTER_SIZES=("2" "3" "7" "11")

cd ../build_omp
./bench c$SLURM_ARRAY_TASK_ID 100 12.4 2 1 0 0 ../../../data_pool/synthetic/pede.bin ../../../data_pool/px_101016/gainMaps_M022.bin ../../../data_pool/synthetic/random_clusters_overlapping/cluster${CLUSTER_SIZES[${SLURM_ARRAY_TASK_ID}]}.bin clustersizes${CLUSTER_SIZES[${SLURM_ARRAY_TASK_ID}]}
