#!/bin/sh

set -x

# run woht
export KMP_AFFINITY="verbose,granularity=fine,proclist=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39],explicit" #,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]"

cd woht_static

for I in 1 2 4 6 8 10 20 40
do
	export OMP_NUM_THREADS=$I
	cd ../woht_static
	./bench 0 10 12.4 1 1 0 0 ../../../data_pool/px_101016/allpede_250us_1243__B_000000.dat ../../../data_pool/px_101016/gainMaps_M022.bin ../../../data_pool/px_101016/Insu_6_tr_1_45d_250us__B_000000.dat cores$I /bigdata/hplsim/production/jungfrau-photoncounter/real/energy.bin /bigdata/hplsim/production/jungfrau-photoncounter/real/photon.bin
done

# run ht
unset KMP_AFFINITY
export KMP_AFFINITY="verbose,compact" #,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]"

for I in 2 4 8 12 16 20 40 80
do
	export OMP_NUM_THREADS=$I
	cd ../ht_static
	./bench 0 10 12.4 1 1 0 0 ../../../data_pool/px_101016/allpede_250us_1243__B_000000.dat ../../../data_pool/px_101016/gainMaps_M022.bin ../../../data_pool/px_101016/Insu_6_tr_1_45d_250us__B_000000.dat cores$I /bigdata/hplsim/production/jungfrau-photoncounter/real/energy.bin /bigdata/hplsim/production/jungfrau-photoncounter/real/photon.bin
done

