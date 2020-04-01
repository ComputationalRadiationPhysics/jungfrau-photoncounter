#!/bin/python

import os
import sys
from itertools import product

# define possible values
masking = [0, 1]
devFrames = [10, 100]#, 1000]
maxValue = [0, 1]
summation = [0, 2, 10, 20, 100]
clusterSizes = [2, 3, 7, 11]
clusterFiles = ["JF", "CL0", "CL4", "CL8"]
energyFiles = ["JF", "G0", "G13"]


# all configurations
#configurations += list(product([0], devFames, clusterSizes, [2], masking, [0], clusterFiles))
#configurations = list(product(summation, devFrames, [2], [0, 1], masking, maxValue, energyFiles))


# base configurations with only 10 frames per device
configurations = list(product([0], [10], [3], [0, 1], [1], [0], energyFiles))
configurations += list(product([0], [10], [3], [2], [1], [0], clusterFiles))

# different sum frames
configurations += list(product([2], [100], [3], [0, 1], [1], [0], energyFiles))
configurations += list(product([10], [100], [3], [0, 1], [1], [0], energyFiles))
configurations += list(product([20], [100], [3], [0, 1], [1], [0], energyFiles))
configurations += list(product([100], [100], [3], [0, 1], [1], [0], energyFiles))

# max value
configurations += list(product([0], [100], [3], [0, 1], [1], [1], energyFiles))

# different cluster sizes
configurations += list(product([0], [100], [3], [2], [1], [0], clusterFiles))
configurations += list(product([0], [100], [7], [2], [1], [0], clusterFiles))
configurations += list(product([0], [100], [11], [2], [1], [0], clusterFiles))

# no masking
configurations += list(product([0], [100], [3], [0, 1], [0], [0], energyFiles))
configurations += list(product([0], [100], [3], [2], [0], [0], clusterFiles))

# remove impossible configurations
configurations = [(s, d, c, m, mask, mv, files) for (s, d, c, m, mask, mv, files) in configurations if s <= d]

# read configuration index
configID = int(sys.argv[1])

# get config
(sumFrames, devFramesOption, clusterSize, mode, maskingEnable, maxValueEnable, inputFile) = configurations[configID]

# initialize values
summationEnable = 0 if sumFrames == 0 else 1
benchmarkID = (max(summation.index(sumFrames) - 1, 0)) * len(devFrames) * len(clusterSizes) + devFrames.index(devFramesOption) * len(clusterSizes) + clusterSizes.index(clusterSize)
    
iterationCount = 100
beamConst = 12.4

# extract file pathes
pedestalPath = ""
gainPath = ""
dataPath = ""

if inputFile == "JF": # jungfrau data
    pedestalPath = "../../../data_pool/px_101016/allpede_250us_1243__B_000000.dat"
    gainPath = "../../../data_pool/px_101016/gainMaps_M022.bin"
    dataPath = "../../../data_pool/px_101016/Insu_6_tr_1_45d_250us__B_000000.dat"
elif inputFile == "G0": # g0 data
    pedestalPath = "../../../data_pool/synthetic/pede.bin"
    gainPath = "../../../data_pool/px_101016/gainMaps_M022.bin"
    dataPath = "../../../data_pool/synthetic/g0.bin"
elif inputFile == "G13": # g13 data
    pedestalPath = "../../../data_pool/synthetic/pede.bin"
    gainPath = "../../../data_pool/px_101016/gainMaps_M022.bin"
    dataPath = "../../../data_pool/synthetic/g13.bin"
elif inputFile == "CL0": # cl0 data
    pedestalPath = "../../../data_pool/synthetic/pede.bin"
    gainPath = "../../../data_pool/px_101016/gainMaps_M022.bin"
    dataPath = "../../../data_pool/synthetic/cluster_0.bin"
elif inputFile == "CL4": # cl4 data
    pedestalPath = "../../../data_pool/synthetic/pede.bin"
    gainPath = "../../../data_pool/px_101016/gainMaps_M022.bin"
    dataPath = "../../../data_pool/synthetic/cluster_4.bin"
elif inputFile == "CL8": # cl8 data
    pedestalPath = "../../../data_pool/synthetic/pede.bin"
    gainPath = "../../../data_pool/px_101016/gainMaps_M022.bin"
    dataPath = "../../../data_pool/synthetic/cluster_8.bin"
else:
    print("error: unexpected parameter")
    abort()

'''
if inputFile == "JF": # jungfrau data
    pedestalPath = "/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/px_101016/allpede_250us_1243__B_000000.dat"
    gainPath = "/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/px_101016/gainMaps_M022.bin"
    dataPath = "/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/px_101016/Insu_6_tr_1_45d_250us__B_000000.dat"
elif inputFile == "G0": # g0 data
    pedestalPath = "/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/pede.bin"
    gainPath = "/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/px_101016/gainMaps_M022.bin"
    dataPath = "/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/g0.bin"
elif inputFile == "G13": # g13 data
    pedestalPath = "/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/pede.bin"
    gainPath = "/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/px_101016/gainMaps_M022.bin"
    dataPath = "/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/g13.bin"
elif inputFile == "CL0": # cl0 data
    pedestalPath = "/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/pede.bin"
    gainPath = "/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/px_101016/gainMaps_M022.bin"
    dataPath = "/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/cluster_0.bin"
elif inputFile == "CL4": # cl4 data
    pedestalPath = "/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/pede.bin"
    gainPath = "/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/px_101016/gainMaps_M022.bin"
    dataPath = "/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/cluster_4.bin"
elif inputFile == "CL8": # cl8 data
    pedestalPath = "/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/pede.bin"
    gainPath = "/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/px_101016/gainMaps_M022.bin"
    dataPath = "/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/cluster_8.bin"
else:
    print("error: unexpected parameter")
    abort()

'''
    
print("./bench " + str(benchmarkID) + " " + str(iterationCount) + " " + str(beamConst) + " " + str(mode) + " " + str(maskingEnable) + " " + str(maxValueEnable) + " " + str(summationEnable) + " " + str(pedestalPath) + " " + str(gainPath) + " " + str(dataPath))
os.system("./bench " + str(benchmarkID) + " " + str(iterationCount) + " " + str(beamConst) + " " + str(mode) + " " + str(maskingEnable) + " " + str(maxValueEnable) + " " + str(summationEnable) + " " + str(pedestalPath) + " " + str(gainPath) + " " + str(dataPath))
