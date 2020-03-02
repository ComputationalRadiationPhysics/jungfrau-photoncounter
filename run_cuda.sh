#!/bin/bash
#SBATCH --job-name=CUDA_Run_1
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem=300000
#SBATCH --cpus-per-task=24
#SBATCH --exclusive
#SBATCH --array=0-256x
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.schenke@hzdr.de
#SBATCH -o log.slurm-cuda_%A_%a.out
#SBATCH -e err.slurm-cuda_%A_%a.out

set -x

export CUDA_VISIBLE_DEVICES=0

export MODULEPATH=$MODULEPATH:/home/schenk24/tools/hemera_spack/spack/share/spack/modules/linux-centos7-x86_64/
module load git gcc cmake cuda boost/1.68.0 alpaka-develop-gcc-6.4.0-iu4hm3r python 

cd build_cuda
python ../run.py $SLURM_ARRAY_TASK_ID
