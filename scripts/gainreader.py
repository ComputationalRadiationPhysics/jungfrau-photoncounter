#!/bin/python

# remove X11 dependecy
import matplotlib as mpl
mpl.use('Agg')

from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
import os, sys

try:
    f = open(sys.argv[1], "rb")
    data = np.fromfile(f, dtype=np.float64).reshape((3, 512, 1024))
    
    for i in range(3):
        output_path = sys.argv[1] + '_' + str(i) + '.bmp'
        plt.figure()
        plt.imshow(data[i], origin='lower')
        plt.colorbar()
        plt.savefig(output_path)
        plt.close()

        print("wrote " + output_path)
           
except:
    print("Error while processing file!")
    print("Exception: " + str(sys.exc_info()[0]))
    pass
