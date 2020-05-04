#!/bin/python

import os
import numpy as np
import sys
import re

# define possible values
devFrames = [10, 100]#, 1000]
maxValue = [0, 1]
summation = [0, 2, 10, 20, 100]
clusterSizes = [2, 3, 7, 11]

def decodeBenchmarkID(benchmarkID):
    cs = int(benchmarkID % len(clusterSizes))
    benchmarkID -= cs
    benchmarkID /= len(clusterSizes)
    df = int(benchmarkID % len(devFrames))
    benchmarkID -= df
    benchmarkID /= len(devFrames)
    sf = int(benchmarkID)

    return (clusterSizes[cs], devFrames[df], summation[sf])

def parseFilename(name):
    det = 1
    if name[0:5] == "det2_":
        det = 2
    elif name[0:5] == "det4_":
        det = 4
    elif name[0:5] == "det8_":
        det = 8
    elif name[0:5] == "det16":
        det = 16
    elif name[0:5] == "det32":
        det = 32
    elif name[0:5] == "det64":
        det = 64

    filename = name

    if det != 1:
        name = name[5:]
    
    benchmarkID = int(re.search(r'\d+', name).group())
    offset = len(str(benchmarkID)) + 1
    name = name[offset:]

    iterationCount = int(re.search(r'\d+', name).group())
    offset = len(str(iterationCount)) + 1
    name = name[offset:]

    mode = int(re.search(r'\d+', name).group())
    offset = len(str(mode)) + 1
    name = name[offset:]

    if mode == 0:
        mode = "Energy"
    elif mode == 1:
        mode = "Photon"
    else:
        mode = "Clustering"

    masking = int(re.search(r'\d+', name).group())
    offset = len(str(masking)) + 1
    name = name[offset:]

    maxValue = int(re.search(r'\d+', name).group())
    offset = len(str(maxValue)) + 1
    name = name[offset:]

    summation = int(re.search(r'\d+', name).group())
    offset = len(str(benchmarkID)) + 1
    name = name[offset:]

    dataset = "JF"
    if  "cluster_0.bin" in name:
        dataset = "CL0"
    elif "cluster_4.bin" in name:
        dataset = "CL4"
    elif "cluster_8.bin" in name:
        dataset = "CL8"
    elif "g0.bin" in name:
        dataset = "G0"
    elif "g13.bin" in name:
        dataset = "G13"

    decodedID = decodeBenchmarkID(benchmarkID)

    return (filename, det, benchmarkID, decodedID[0], decodedID[1], decodedID[2], mode, masking, maxValue, summation, dataset)

def getCsvHeader():
    return "Filename, Detector Count, Benchmark ID, Cluster Size, Frames on each Device, Summation, Mode, Masking, Maximum Value enabled, Summation enabled, Dataset, Average time (in s), Standard Deviation from Average time\n"

def getCsvLineFromData(data):
    string = ""
    for elem in data:
        string += str(elem) + ", "
    return string + "\n"

directory = sys.argv[1]

output_path = sys.argv[2]

files = [f for f in os.listdir(directory) if "data_pool" in f]

with open(output_path, "w") as output_file:
    output_file.write(getCsvHeader())
    for f in files:
        parsed = parseFilename(f)
        times = np.loadtxt(directory + "/" + f)
        times = times[10:]
        avg = np.mean(times / 1000000000)
        stddev = np.std(times / 1000000000)
        parsed += (avg, stddev)
        
        output_file.write(getCsvLineFromData(parsed))

    output_file.flush()
