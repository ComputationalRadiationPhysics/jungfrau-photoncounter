#! /bin/bash
#SBATCH --job-name=CUDA_Compile
#SBATCH --partition=gpu
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=25G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.schenke@hzdr.de
#SBATCH -o log.cc.slurm-%j.out
#SBATCH -e err.cc.slurm-%j.out

set -x

# load modules
export alpaka_DIR=/home/schenk24/workspace/alpaka/
module load git gcc cmake cuda boost python

# build for 1 gpu
mkdir -p build_cuda_1
cd build_cuda_1
cmake .. -DCMAKE_BUILD_TYPE=Release -DALPAKA_CUDA_ARCH=60 -DBENCHMARKING_ENABLED=ON
make -j
cd ..

# build for 2 gpu
mkdir -p build_cuda_2
cd build_cuda_2
cmake .. -DCMAKE_BUILD_TYPE=Release -DALPAKA_CUDA_ARCH=60 -DBENCHMARKING_ENABLED=ON
make -j
cd ..

# build for 4 gpu
mkdir -p build_cuda_4
cd build_cuda_4
cmake .. -DCMAKE_BUILD_TYPE=Release -DALPAKA_CUDA_ARCH=60 -DBENCHMARKING_ENABLED=ON
make -j
cd ..

