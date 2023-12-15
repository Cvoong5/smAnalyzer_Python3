import numpy as np
import matplotlib.pyplot as plt
<<<<<<< HEAD
from matplotlib.patches import Circle
from scipy.optimize import curve_fit
import time

#Function to generate a 2D Gaussian
def gauss2d(dimension, normalization, intensity, y_center, x_center, y_sd, x_sd):
    y , x = dimension
    return intensity + normalization * np.exp( -(1/2) * ( (x-x_center)**2 / x_sd + (y-y_center)**2 / y_sd ))
=======
import scipy.optimize as opt

#Function to generate a 2D Gaussian
def gauss2d(xy = (512, 512), xo = 0, yo = 0, sx = 1, sy = 1):
    #Set up to generate a single molecule based on 2D Gaussian
    x, y = xy
    normalization = 1/(2*np.pi*sx*sy)
    x_term = ((x-xo)**2/sx**2)
    y_term = ((y-yo)**2/sy**2)
    gauss2d = np.exp(-(1/2)*(x_term + y_term))
    return normalization*gauss2d
>>>>>>> fdf71a2d66415fa302371599c8b2dad504082c91
#Parameters
def generate_random_molecules(num_molecules = 100, image_size = (512, 512)):
# Create an empty 512x512 image array 
    image = np.zeros(image_size)
    x, y = np.meshgrid(np.arange(image_size[1]), np.arange(image_size[0]))

#Perform matrix operations to randomize spot localization
    for _ in range(num_molecules):
        xo = np.random.randint(0, image_size[1])
        yo = np.random.randint(0, image_size[0])
<<<<<<< HEAD
        image += gauss2d([y, x], 1, 0, xo, yo, 1, 1)
    return image

#Generate single molecules
molecules = generate_random_molecules()
row, col = molecules.shape
initial_coordinates = []
counter = 0
for y in range(10, row - 10):
    for x in range(10, col - 10):
=======
        image += gauss2d((x,y), xo, yo)
    return image

#Needs some work
def gauss2dfit(data):
    rows, cols = data.shape
    empty_data = np.zeros_like(data)
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    intensity_max = data.argmax()
    guess = [x.ravel()[intensity_max], y.ravel()[intensity_max], 1, 1]
    parameters, covariance = opt.curve_fit(gauss2d, empty_data, data, p0 = guess)
    return 
>>>>>>> fdf71a2d66415fa302371599c8b2dad504082c91

