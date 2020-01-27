#!/bin/python

import sys
import operator
import numpy as np

## remove X11 dependecy
#import matplotlib as mpl
#mpl.use('Agg')

from matplotlib import pyplot as plt
from matplotlib import colors

numbers = []
clusters = []

with open(sys.argv[1]) as f:
    for line in f:
        numbers += [float(x) for x in line.split()] 

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
    clusters += [(frameNumber, x, y, cluster)]

unsortedFrameNumbers = [x[0] for x in clusters]
frameNumbers = sorted({x:unsortedFrameNumbers.count(x) for x in unsortedFrameNumbers}.items(), key=operator.itemgetter(0))



print(frameNumbers)
abort()




clusters = sorted(clusters, key=lambda t: t[0])

print(str(cluster_count) + " clusters found!")

i = 0
for c in clusters:
    data = np.zeros(shape=(512, 1024))

    for i in range(frameNumbers[0][1]):
        (_, pos_x, pos_y, cluster) = clusters[i]
        cluster = np.asarray(cluster)

        for x in range(3):
            for y in range(3):
                data[y - 1 + pos_y, x - 1 + pos_x] = cluster[y, x]

    try:
        print("trying log scale")
        plt.figure()
        plt.imshow(data, norm=colors.LogNorm(), origin='lower')
        plt.colorbar()
        #plt.show()
        plt.savefig("clusters_log_" + str(i) + ".png")
        plt.close()
    except:
        print("falling back to linear representation")
        plt.figure()
        plt.imshow(data, origin='lower')
        plt.colorbar()
        #plt.show()
        plt.savefig("clusters_lin_" + str(i) + ".png")
        plt.close()
        pass
    


'''
for fn in frameNumbers:
    print(fn)
    
for c in clusters:
    print(c)
'''
