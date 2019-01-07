#!/bin/python

import sys
import operator
import numpy as np
import math
from collections import namedtuple
from itertools import groupby

Cluster = namedtuple("Cluster", "frameNumber coord_x coord_y clusterValues")

distanceThreshold = 10

interesting_frames = [x for x in range(500)] + [x for x in range(9090, 9100)]
highest_interesting_frame = 7792170 + 9100

def loadClusters(path, maxframe):
    numbers = []
    clusters = []

    i = 1

    with open(path) as f:
        for line in f:
            numbers += [int(x) for x in line.split()]

            while i + (1 + 1 + 1 + 9) < len(numbers):
                frameNumber = numbers[i]
                i += 1
                x = numbers[i]
                i += 1
                y = numbers[i]
                i += 1
                cluster = [[numbers[i + y * 3 + x] for x in range(3)] for y in range(3)]
                i += 9
                clusters += [Cluster(frameNumber, x, y, cluster)]        

            if(len(clusters) > 0):
                if clusters[-1].frameNumber >= maxframe:
                    break;

    cluster_count = numbers[0]
        
    print(str(cluster_count) + " clusters found!")

    return clusters

def squaredEuclideanDistance(vec1, vec2):
    result = 0
    
    if(len(vec1) is not len(vec2)):
        raise ValueError('Dimensions not equal')
    
    for i in range(len(vec1)):
        result += (vec1[i] - vec2[i])**2

    return result
    
def getClusterValueDifference(cluster1, cluster2):
    return squaredEuclideanDistance(cluster1.clusterValues, cluster2.clusterValues)

def getClusterDistance(cluster1, cluster2):
    return squaredEuclideanDistance([cluster1.coord_x, cluster1.coord_y], [cluster2.coord_x, cluster2.coord_y])

def isNearbyCluster(cluster1, cluster2):
    if(getClusterDistance(cluster1, cluster2) > 10):
        return True
    else:
        return False
    
clusters_detector = loadClusters(sys.argv[1], highest_interesting_frame)
clusters_detector.sort(key=lambda x: x.frameNumber)
clusters_detector = [(i, [k for k in j]) for i,j in groupby(clusters_detector, key=lambda x: x.frameNumber)]

print("loaded detector clusters")

clusters_reference = loadClusters(sys.argv[2], highest_interesting_frame)
clusters_reference.sort(key=lambda x: x.frameNumber)
clusters_reference = [(i, [k for k in j]) for i, j in groupby(clusters_reference, key=lambda x: x.frameNumber)]

print("loaded reference clusters")

frames_detector = np.array([i for i, j in  clusters_detector])
frames_reference = np.array([i for i, j in  clusters_reference])

missing_frames_detector = np.setdiff1d(frames_reference, frames_detector)
missing_frames_reference = np.setdiff1d(frames_detector, frames_reference)

if(len(missing_frames_detector)):
    print("WARNING: The detector data set includes the frames", missing_frames_detector, "which are not present in the reference data set!")
 
if(len(missing_frames_reference)):
    print("WARNING: The reference data set includes the frames", missing_frames_reference, "which are not present in the detector data set!")

frameNumbers = np.intersect1d(frames_reference, frames_detector)
clusters_detector = [i for i in clusters_detector if i[0] in frameNumbers]
clusters_reference = [i for i in clusters_reference if i[0] in frameNumbers]

for framenumber in interesting_frames:
    frame = np.zeros((401, 401))
    for cluster in clusters_detector[framenumber][1]:
        x = cluster.coord_x - 1
        y = cluster.coord_y - 1
        for xpos in range(3):
            for ypos in range(3):
                if frame[x + xpos, y + ypos] == 0:
                    frame[x + xpos, y + ypos] = 1

    for cluster in clusters_reference[framenumber][1]:
        x = cluster.coord_x - 1
        y = cluster.coord_y - 1
        for xpos in range(3):
            for ypos in range(3):
                if frame[x + xpos, y + ypos] < 2:
                    frame[x + xpos, y + ypos] = frame[x + xpos, y + ypos] + 2
                    
    np.savetxt("frame_"+str(framenumber)+".txt", frame)
