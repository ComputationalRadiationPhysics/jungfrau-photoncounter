#! /bin/bash
#SBATCH --job-name=CUDA_Compile
#SBATCH --partition=fwkt_v100
#SBATCH -A fwkt_v100
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
export alpaka_DIR=/home/schenk24/workspace/alpaka/install/
module load git gcc cmake cuda boost python

cd ..

# build for 1 gpu
mkdir -p build_cuda_1
cd build_cuda_1
cmake .. -DCMAKE_BUILD_TYPE=Release -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE=ON -DALPAKA_ACC_GPU_CUDA_ENABLE=ON -DALPAKA_CUDA_ARCH=60 -DBENCHMARKING_ENABLED=ON
cmake --build . --verbose > build_log.txt
cd ..

# build for 2 gpu
mkdir -p build_cuda_2
cd build_cuda_2
cmake .. -DCMAKE_BUILD_TYPE=Release -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE=ON -DALPAKA_ACC_GPU_CUDA_ENABLE=ON -DALPAKA_CUDA_ARCH=60 -DCMAKE_C_FLAGS_RELEASE="-O3 --use_fast_math" -DBENCHMARKING_ENABLED=ON
cmake --build . --verbose > build_log.txt
cd ..

# build for 3 gpu
mkdir -p build_cuda_3
cd build_cuda_3
cmake .. -DCMAKE_BUILD_TYPE=Release -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE=ON -DALPAKA_ACC_GPU_CUDA_ENABLE=ON -DALPAKA_CUDA_ARCH=60 -DCMAKE_C_FLAGS_RELEASE="-O3 --use_fast_math" -DBENCHMARKING_ENABLED=ON
cmake --build . --verbose > build_log.txt
cd ..


# build for 4 gpu
mkdir -p build_cuda_4
cd build_cuda_4
cmake .. -DCMAKE_BUILD_TYPE=Release -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE=ON -DALPAKA_ACC_GPU_CUDA_ENABLE=ON -DALPAKA_CUDA_ARCH=60 -DCMAKE_C_FLAGS_RELEASE="-O3 --use_fast_math" -DBENCHMARKING_ENABLED=ON
cmake --build . --verbose > build_log.txt
cd ..

