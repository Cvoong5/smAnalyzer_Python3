#smSupport.py
import numpy as np
import pandas as pd
import os
import smSupport as smS
import smCore as smC
import smColocalize as sm2
import matplotlib.pyplot as plt


#Read files

directory = '/Users/calvin/Desktop/smF_analyzer/analysis'
os.chdir(directory)
file1 = 'Green002.tif'
file2 = 'Red002.tif'

movie = smC.read_movie(file1)
movie2 = smC.read_movie(file2)
summed_image = smC.sum_movie(movie)
summed_image2 = smC.sum_movie(movie2)
flipped_image = smS.offset_image(summed_image2, angle = 10, x_offset = -5, flip = "h")

sm2.overlay_images(summed_image, flipped_image)

#coordinates1 = smC.find_spots(image1, 3)
#ocoordinates1 = smS.organize_coordinates(coordinates1)
#movie2, image2 = smC.read_data(file2)
#coordinates2 = smC.find_spots(image2, 1)
#ocoordinates1 = smS.organize_coordinates(coordinates2)


