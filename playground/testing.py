# testing.py

# this code performs some iterative tests on the pixel frustum simulation in occlusion.py

import random
import string
import matplotlib.pyplot as plt
import matplotlib.collections
import numpy as np
import math
from sys import float_info

from occlusion import simulate

numRays = 1000
wronglyCulledRays = []
particleOffsets = np.arange(0, 3, 0.1)
samples = 16

for pOff in particleOffsets:

    print(str(pOff) + " ... ")

    sum = 0
    for i in range(samples):
        sum += simulate(num_rays=numRays, seed=i, plot=False, particle_voxel_offset=pOff, log=False)
    sum /= samples

    wronglyCulledRays.append(sum / numRays)

plt.plot(particleOffsets, wronglyCulledRays)
plt.show()