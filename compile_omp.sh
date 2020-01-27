#! /bin/bash
#SBATCH --job-name=OpenMP_Compile
#SBATCH --partition=intel_32
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.schenke@hzdr.de
#SBATCH -o log.slurm-%j.out
#SBATCH -e err.slurm-%j.out

set -x

export MODULEPATH=$MODULEPATH:/home/schenk24/tools/hemera_spack/spack/share/spack/modules/linux-centos7-x86_64/
module load git gcc cmake boost/1.68.0 alpaka-develop-gcc-6.4.0-iu4hm3r 

mkdir build_omp
cd build_omp
cmake .. -DCMAKE_BUILD_TYPE=Release -DBENCHMARKING_ENABLED=ON -DALPAKA_ACC_GPU_CUDA_ENABLE=OFF
make -j
