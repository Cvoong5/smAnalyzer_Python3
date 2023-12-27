from smSimulation import (gauss2d,
                          generate_random_molecules,
                          generate_random_coordinates,
                          detect_molecules,
                          generate_dynamics)
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import os


w_dir = "/Users/calvin/smAnalyzer/smSimulation"
os.chdir(w_dir)
fname = "test_movie.tif"
file = tifffile.imread(fname)
summed_data = np.sum(file, axis = 0)

#==Work on local detection of single molecules
#===Methods to appropriately calculate background signal
#====Draw circles around centroid of single molecules

row, col = summed_data.shape
window = 50
bg_sub = summed_data.copy()
background = np.zeros_like(summed_data)
build = np.zeros_like(summed_data)
background_sub = np.zeros_like(summed_data)
for y in range(window, row - window, window):
    for x in range(window, col - window):
        data = summed_data[y - window: y + window, x - window: x + window]
        background[y - window: y + window, x - window: x + window] = np.mean(data)
        build[y - window: y + window, x - window: x + window] = data
        background_sub[y - window: y + window, x - window: x + window] = data - np.mean(data)

