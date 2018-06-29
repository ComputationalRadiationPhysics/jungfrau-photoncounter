from matplotlib import pyplot as plt
from matplotlib import colors
import sys
import os, glob
import numpy as np

fileNames = glob.glob("*.txt")
print(fileNames)

for f in fileNames:
    data = np.loadtxt(f).reshape((512, 1024))
    (name, _) = os.path.splitext(f)
    print(f)
    plt.imshow(data, norm=colors.LogNorm(), origin='lower')
    plt.colorbar()
    plt.savefig('log/' + name + '.png')
    plt.close()
    
    '''
    plt.imshow(data, origin='lower')
    plt.colorbar()
    plt.savefig('linear/' + name + '.png')
    plt.close()
    '''
