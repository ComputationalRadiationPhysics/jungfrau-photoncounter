#!/bin/bash
#SBATCH --job-name=ClusterCountCpu
#SBATCH --partition=defq
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200000
#SBATCH --cpus-per-task=80
#SBATCH --exclusive
#SBATCH --array=0-8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.schenke@hzdr.de
#SBATCH -o log.intel_clustercount.slurm-%A_%a.out
#SBATCH -e err.intel_clustercount.slurm-%A_%a.out

set -x

# load modules
export alpaka_DIR=/home/schenk24/workspace/alpaka/
module load git intel cmake boost python

export KMP_AFFINITY="verbose,compact"

cd ../build_omp
./bench 0 100 12.4 2 1 0 0 ../../../data_pool/synthetic/pede.bin ../../../data_pool/px_101016/gainMaps_M022.bin ../../../data_pool/synthetic/random_clusters_overlapping/cluster_$SLURM_ARRAY_TASK_ID.bin clustercount$SLURM_ARRAY_TASK_ID
#./bench 5 10 12.4 2 1 0 0 ../../../data_pool/synthetic/pede.bin ../../../data_pool/px_101016/gainMaps_M022.bin ../../../data_pool/synthetic/random_clusters_overlapping/cluster_$SLURM_ARRAY_TASK_ID.bin clustercount$SLURM_ARRAY_TASK_ID
