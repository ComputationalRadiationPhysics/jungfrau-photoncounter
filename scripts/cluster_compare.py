#!/bin/python

import sys
import operator
import numpy as np
import math
from collections import namedtuple
from itertools import groupby

Cluster = namedtuple("Cluster", "frameNumber coord_x coord_y clusterValues")

distanceThreshold = 10
    
def loadClusters(path):
    numbers = []
    clusters = []
    
    with open(path) as f:
        for line in f:
            numbers += [int(x) for x in line.split()] 

    cluster_count = numbers[0]

    i = 1
    while i < len(numbers):
        frameNumber = numbers[i]
        i += 1
        x = numbers[i]
        i += 1
        y = numbers[i]
        i += 1
        cluster = [[numbers[i + y * 3 + x] for x in range(3)] for y in range(3)]
        i += 9
        clusters += [Cluster(frameNumber, x, y, cluster)]
        
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
    
clusters_detector = loadClusters(sys.argv[1])
clusters_detector.sort(key=lambda x: x.frameNumber)
clusters_detector = [(i, [k for k in j]) for i,j in groupby(clusters_detector, key=lambda x: x.frameNumber)]

clusters_reference = loadClusters(sys.argv[2])
clusters_reference.sort(key=lambda x: x.frameNumber)
clusters_reference = [(i, [k for k in j]) for i, j in groupby(clusters_reference, key=lambda x: x.frameNumber)]

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

exact_matches = []
overlap_matches = []
det_only = []
ref_only = []

for i in range(len(clusters_detector)):
    current_det_only = []
    current_ref_only = []
    current_exact_matches = []
    current_overlap_matches = []

    # find exact matches and overlaps
    for cls_det in clusters_detector[i][1]:
        
        for cls_ref in clusters_reference[i][1]:
            if cls_ref.coord_x == cls_det.coord_x and cls_ref.coord_y == cls_det.coord_y:
                current_exact_matches += [(cls_det, cls_ref)]
                
            if squaredEuclideanDistance([cls_ref.coord_x, cls_ref.coord_y], [cls_det.coord_x, cls_det.coord_y]) < 2:
                current_overlap_matches += [(cls_det, cls_ref)]


    # check which clusters could not be matched
    current_det_only = [i for i in clusters_detector[i][1] if i not in [j for j, k in current_overlap_matches]]
    current_ref_only = [i for i in clusters_reference[i][1] if i not in [k for j, k in current_overlap_matches]]

    
    exact_matches += [current_exact_matches]
    overlap_matches += [current_overlap_matches]
    det_only += [current_det_only]
    ref_only += [current_ref_only]
        
num_exact_matches = [len(i) for i in exact_matches]
num_overlap_matches = [len(i) for i in overlap_matches]
num_det_only = [len(i) for i in det_only]
num_ref_only = [len(i) for i in ref_only]

stats = np.array(list(zip(num_exact_matches, num_overlap_matches, num_det_only, num_ref_only)))
np.savetxt("stats.txt", stats)
