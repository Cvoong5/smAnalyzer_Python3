import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

#Function to generate a 2D Gaussian
def gauss2d(x = 0, y = 0, xo = 0, yo = 0, sx = 1, sy = 1):
    #Set up to generate a single molecule based on 2D Gaussian
    normalization = 1/(2*np.pi*sx*sy)
    x_term = ((x-xo)**2/sx**2)
    y_term = ((y-yo)**2/sy**2)
    gauss2d = np.exp(-(1/2)*(x_term + y_term))
    return normalization*gauss2d
#Parameters
def generate_random_molecules(num_molecules = 100, image_size = (512, 512)):
# Create an empty 512x512 image array 
    image = np.zeros(image_size)
    x, y = np.meshgrid(np.arange(image_size[1]), np.arange(image_size[0]))

#Perform matrix operations to randomize spot localization
    for _ in range(num_molecules):
        xo = np.random.randint(0, image_size[1])
        yo = np.random.randint(0, image_size[0])
        image += gauss2d(x, y, xo, yo)
    return image

def local_maxima_2d():
    pass
