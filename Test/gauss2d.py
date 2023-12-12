import numpy as np
import matplotlib.pyplot as plt
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
#Parameters
def generate_random_molecules(num_molecules = 100, image_size = (512, 512)):
# Create an empty 512x512 image array 
    image = np.zeros(image_size)
    x, y = np.meshgrid(np.arange(image_size[1]), np.arange(image_size[0]))

#Perform matrix operations to randomize spot localization
    for _ in range(num_molecules):
        xo = np.random.randint(0, image_size[1])
        yo = np.random.randint(0, image_size[0])
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

