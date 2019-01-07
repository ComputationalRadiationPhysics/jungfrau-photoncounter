#!/bin/python

import numpy as np

gain0 = np.ones((400, 400))
gain1 = np.zeros((400, 400))
gain2 = np.zeros((400, 400))

gain = np.concatenate((gain0, gain1, gain2))

gain.tofile("moench_gain.bin")
