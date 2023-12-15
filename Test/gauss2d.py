import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.optimize import curve_fit
import time

#Function to generate a 2D Gaussian
def gauss2d(dimension, normalization, intensity, y_center, x_center, y_sd, x_sd):
    y , x = dimension
    return intensity + normalization * np.exp( -(1/2) * ( (x-x_center)**2 / x_sd + (y-y_center)**2 / y_sd ))
#Parameters
def generate_random_molecules(num_molecules = 100, image_size = (512, 512)):
# Create an empty 512x512 image array 
    image = np.zeros(image_size)
    x, y = np.meshgrid(np.arange(image_size[1]), np.arange(image_size[0]))

#Perform matrix operations to randomize spot localization
    for _ in range(num_molecules):
        xo = np.random.randint(0, image_size[1])
        yo = np.random.randint(0, image_size[0])
        image += gauss2d([y, x], 1, 0, xo, yo, 1, 1)
    return image

#Generate single molecules
molecules = generate_random_molecules()
row, col = molecules.shape
initial_coordinates = []
counter = 0
for y in range(10, row - 10):
    for x in range(10, col - 10):

