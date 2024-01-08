import sma
import os
import matplotlib.pyplot as plt
import numpy as np

file = "/Users/calvin/Desktop/smF_analyzer/20240104/GADD45_Cy3_Normal_ND3.nd2" 

movie, summed_image = sma.imread(file)
bg_sub, background = sma.TemporalMedianFilter(summed_image, radius = 3)
coordinates = sma.detect(bg_sub, threshold = 1500)
fig, ax = plt.subplots()
ax.imshow(bg_sub)
for coord in coordinates:
    y, x = coord
    circ = plt.Circle((x,y), fill = None, radius = 3, color = "red")
    ax.add_patch(circ)
    plt.draw()
    plt.pause(0.06)
plt.show()

