#!/bin/sh
#SBATCH --job-name=omp_check
#SBATCH --partition=defq
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=150000
#SBATCH --cpus-per-task=40
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.schenke@hzdr.de
#SBATCH -o log.ob.slurm-%A_%a.out
#SBATCH -e err.ob.slurm-%A_%a.out

set -x

export alpaka_DIR=/home/schenk24/workspace/alpaka/install/
module load git intel cmake boost python
export CC=icc
export CXX=icpc

# run ht
unset KMP_AFFINITY
export KMP_AFFINITY="verbose,compact"

cd ..
mkdir -p build_omp_test
cd build_omp_test
cmake .. -DCMAKE_BUILD_TYPE=Release -DBENCHMARKING_ENABLED=ON -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE=ON -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE=ON -DALPAKA_ACC_GPU_CUDA_ENABLE=OFF -DCMAKE_C_FLAGS_RELEASE="-O3 -march=native -qopt-zmm-usage=high -fp-model precise -DNDEBUG" -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native  -qopt-zmm-usage=high -fp-model precise -DNDEBUG"
cmake --build . --verbose > build_log.txt

export OMP_NUM_THREADS=40
echo "===== OMP ====="
echo "----- Energy -----"
./bench 0 3 12.4 0 1 1 1 ../../../data_pool/px_101016/allpede_250us_1243__B_000000.dat ../../../data_pool/px_101016/gainMaps_M022.bin ../../../data_pool/px_101016/Insu_6_tr_1_45d_250us__B_000000.dat corestest /bigdata/hplsim/production/jungfrau-photoncounter/reference/energy.bin /bigdata/hplsim/production/jungfrau-photoncounter/reference/bin/photon.bin /bigdata/hplsim/production/jungfrau-photoncounter/reference/maxValues.bin /bigdata/hplsim/production/jungfrau-photoncounter/reference/sum.bin /bigdata/hplsim/production/jungfrau-photoncounter/reference/clusters.bin
echo "----- Photon -----"
./bench 0 3 12.4 1 1 0 0 ../../../data_pool/px_101016/allpede_250us_1243__B_000000.dat ../../../data_pool/px_101016/gainMaps_M022.bin ../../../data_pool/px_101016/Insu_6_tr_1_45d_250us__B_000000.dat corestest /bigdata/hplsim/production/jungfrau-photoncounter/reference/energy.bin /bigdata/hplsim/production/jungfrau-photoncounter/reference/bin/photon.bin /bigdata/hplsim/production/jungfrau-photoncounter/reference/maxValues.bin /bigdata/hplsim/production/jungfrau-photoncounter/reference/sum.bin /bigdata/hplsim/production/jungfrau-photoncounter/reference/clusters.bin
echo "----- Clustering -----"
./bench 0 3 12.4 3 1 0 0 ../../../data_pool/px_101016/allpede_250us_1243__B_000000.dat ../../../data_pool/px_101016/gainMaps_M022.bin ../../../data_pool/px_101016/Insu_6_tr_1_45d_250us__B_000000.dat corestest /bigdata/hplsim/production/jungfrau-photoncounter/reference/cluster_energy.bin /bigdata/hplsim/production/jungfrau-photoncounter/reference/bin/photon.bin /bigdata/hplsim/production/jungfrau-photoncounter/reference/maxValues.bin /bigdata/hplsim/production/jungfrau-photoncounter/reference/sum.bin /bigdata/hplsim/production/jungfrau-photoncounter/reference/clusters.bin

