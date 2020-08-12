#!/bin/sh

set -x

#cd icc_build
#cd ../icc_serial
#./bench 0 10 12.4 1 1 0 0 ../../../data_pool/px_101016/allpede_250us_1243__B_000000.dat ../../../data_pool/px_101016/gainMaps_M022.bin ../../../data_pool/px_101016/Insu_6_tr_1_45d_250us__B_000000.dat cores1 /bigdata/hplsim/production/jungfrau-photoncounter/real/energy.bin /bigdata/hplsim/production/jungfrau-photoncounter/real/photon.bin
#cp cores12_0_10_1_1_0_0_..\ ..\ ..\ data_pool\ px_101016 allpede_250us_1243__B_000000.dat_..\ ..\ ..\ data_pool\ px_101016\ gainMaps_M022.bin_..\ ..\ ..\ data_pool\ px_101016\ Insu_6_tr_1_45d_250us__B_000000.dat.txt ../icc_build/
#cd ../icc_build/


export KMP_AFFINITY="verbose,granularity=fine,proclist=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],explicit" #,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]"

for I in 1 2 4 6 8 10 20
do
	export OMP_NUM_THREADS=$I
	./bench 0 10 12.4 1 1 0 0 ../../../data_pool/px_101016/allpede_250us_1243__B_000000.dat ../../../data_pool/px_101016/gainMaps_M022.bin ../../../data_pool/px_101016/Insu_6_tr_1_45d_250us__B_000000.dat cores$I /bigdata/hplsim/production/jungfrau-photoncounter/real/energy.bin /bigdata/hplsim/production/jungfrau-photoncounter/real/photon.bin
done

python ../read_cores.py
