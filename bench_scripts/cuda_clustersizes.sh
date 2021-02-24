#!/bin/bash
#SBATCH --job-name=ClusterCountGpu
#SBATCH --partition=fwkt_v100
#SBATCH -A fwkt_v100
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem=370G
#SBATCH --cpus-per-task=24
#SBATCH --exclusive
#SBATCH --array=0-3
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.schenke@hzdr.de
#SBATCH -o log.ob.slurm-%A_%a.out
#SBATCH -e err.ob.slurm-%A_%a.out

set -x

# load modules
export alpaka_DIR=/home/schenk24/workspace/alpaka/
module load git gcc cuda cmake boost python

export CUDA_VISIBLE_DEVICES=0

export GOMP_CPU_AFFINITY=0-11
export OMP_PROC_BIND=true

CLUSTER_SIZES=("2" "3" "7" "11")

cd ../build_cuda_1
./bench c$SLURM_ARRAY_TASK_ID 100 12.4 2 1 0 0 ../../../data_pool/synthetic/pede_cl${CLUSTER_SIZES[${SLURM_ARRAY_TASK_ID}]}.bin ../../../data_pool/px_101016/gainMaps_M022.bin ../../../data_pool/synthetic/random_clusters_overlapping/cluster${CLUSTER_SIZES[${SLURM_ARRAY_TASK_ID}]}.bin clustersizes${CLUSTER_SIZES[${SLURM_ARRAY_TASK_ID}]}
