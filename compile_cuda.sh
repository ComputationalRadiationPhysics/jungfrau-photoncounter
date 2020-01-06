#! /bin/bash
#SBATCH --job-name=CUDA_Compile
#SBATCH --partition=gpu
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.schenke@hzdr.de
#SBATCH -o log.slurm-%j.out
#SBATCH -e err.slurm-%j.out

set -x

export MODULEPATH=$MODULEPATH:/home/schenk24/tools/hemera_spack/spack/share/spack/modules/linux-centos7-x86_64/
module load git gcc cmake cuda boost/1.68.0 alpaka-develop-gcc-6.4.0-iu4hm3r 

mkdir build_cuda
cd build_cuda
cmake .. -DCMAKE_BUILD_TYPE=Release -DALPAKA_CUDA_ARCH=60 -DBENCHMARKING_ENABLED=ON
make -j
