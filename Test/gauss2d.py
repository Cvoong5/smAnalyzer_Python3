import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.optimize import curve_fit
import time

#Function to generate a 2D Gaussian
def gauss2d(dimension, normalization, intensity, y_center, x_center, y_sd, x_sd):
    y , x = dimension
    return intensity + normalization * np.exp( -(1/2) * ( (x-x_center)**2 / x_sd + (y-y_center)**2 / y_sd ))

#Function to generate a 2D Gaussian
def generate_random_array(num_array = 100, image_size = (512, 512)):
# Create an empty 512x512 image array 
    image = np.zeros(image_size)
    x, y = np.meshgrid(np.arange(image_size[1]), np.arange(image_size[0]))

#Perform matrix operations to randomize spot localization
    for _ in range(num_array):
        xo = np.random.randint(0, image_size[1])
        yo = np.random.randint(0, image_size[0])
        image += gauss2d([y, x], 1, 0, xo, yo, 1, 1)
    return image

#Generate single array
def local_maxima(array):
    row, col = array.shape
    coordinates = []
    for y in range(1, row - 1):
        for x in range(1, col - 1):
            xlow = array[y, x - 1]
            xmid = array[y, x]
            xhigh = array[y, x + 1]
            ylow = array[y - 1, x]
            ymid = array[y, x]
            yhigh = array[y + 1, x]
            if xmid > xlow:
                if xmid > xhigh:
                    if ymid > ylow:
                        if ymid > yhigh:
                            print(f"coordinate saved {y, x}")
                            coordinates.append([y, x])
                        if ymid == yhigh:
                            for y2 in range(row - 1):
                                y2mid = array[y2, x]
                                y2high = array[y2 + 1, x]
                                if y2mid > y2high:
                                    coordinates.append([y2, x])

                if xmid == xhigh:
                    for x2 in range(col - 1):
                        x2mid = array[y, x2]
                        x2high = array[y, x2 + 1]
                        if x2mid > x2high:
                            if ymid > ylow:
                                if ymid > yhigh:
                                    print(f"coordinate saved {y, x}")
                                    coordinates.append([y, x])
                                if ymid == yhigh:
                                    for y3 in range(row - 1):
                                        y3mid = array[y2, x]
                                        y3high = array[y2 + 1, x]
                                        if y3mid > y3high:
                                            coordinates.append([y3, x])
    return coordinates

molecules = generate_random_array()
coordinates_estimate = local_maxima(molecules)

for coord in coordinates_estimate:
    yo, xo = coord
    image = molecules[yo-6:yo+6, xo-6:xo+6]
    reference = np.zeros_like(image)
    normalization = 1
    intensity = 0
    y_sd = 1
    x_sd = 1
    guess = [normalization, intensity, yo, xo, y_sd, x_sd]
    if image.shape == (12,12):
        curve_fit(gauss2d, reference, image, p0 = guess) #Failing
    else:
        pass
