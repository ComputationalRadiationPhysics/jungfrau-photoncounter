#!/bin/bash
#SBATCH --job-name=CUDA_Run_base_2
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem=350000
#SBATCH --cpus-per-task=24
#SBATCH --exclusive
#SBATCH --array=0-9
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.schenke@hzdr.de
#SBATCH -o log.cb2.slurm-%A_%a.out
#SBATCH -e err.cb2.slurm-%A_%a.out

set -x

# load modules
export alpaka_DIR=/home/schenk24/workspace/alpaka/
module load git gcc cmake cuda boost python

cd build_cuda_2
python ../run_base.py $SLURM_ARRAY_TASK_ID
