#!/bin/python

# remove X11 dependecy
#import matplotlib as mpl
#mpl.use('Agg')

from matplotlib import pyplot as plt
from matplotlib import colors
import sys
import os, glob
import numpy as np

fileNames = glob.glob("*.txt")
print(fileNames)

for f in fileNames:
    try:
        print(f)
        data = np.loadtxt(f).reshape((512, 1024))
        (name, _) = os.path.splitext(f)

        try:
            plt.figure()
            plt.imshow(data, norm=colors.LogNorm(), origin='lower')
            plt.colorbar()
            plt.savefig('log/' + name + '.png')
            plt.close()
        except:
            print("Generation of log/" + name + ".png failed!")
            pass
        
        try:
            plt.figure()
            plt.imshow(data, origin='lower')
            plt.colorbar()
            plt.savefig('linear/' + name + '.png')
            plt.close()
        except:
            print("Generation of linear/" + name + ".png failed!")
            pass

    except:
        print("Something with the file " + f + " went wrong!")
        pass
        
