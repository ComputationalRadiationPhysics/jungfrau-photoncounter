#! /bin/bash
#SBATCH --job-name=OpenMP_Compile
#SBATCH --partition=defq
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=15g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.schenke@hzdr.de
#SBATCH -o log.oc.slurm-%j.out
#SBATCH -e err.oc.slurm-%j.out

set -x

export alpaka_DIR=/home/schenk24/workspace/alpaka/install/
module load git intel cmake boost python

mkdir -p build_omp
cd build_omp
cmake .. -DCMAKE_BUILD_TYPE=Release -DBENCHMARKING_ENABLED=ON -DALPAKA_ACC_GPU_CUDA_ENABLE=OFF -DCMAKE_C_FLAGS_RELEASE="-O3 -march=native -DNDEBUG" -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native -DNDEBUG"
make -j
