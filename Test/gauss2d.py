import numpy as np
import matplotlib.pyplot as plt

#Function to generate a 2D Gaussian
def gauss2d(x = 0, y = 0, xo = 0, yo = 0, sx = 1, sy = 1):
    normalization = 1/(2*np.pi*sx*sy)
    gauss2d = np.exp(-(((x-xo)**2/sx**2) + ((y-yo)**2)/(sy**2))/2)
    return normalization*gauss2d

#Parameters
image_size = 512
num_molecules = 100

# Create an empty 512x512 image array 
image = np.zeros((512,512))
# Generate a grid
x, y = np.meshgrid(np.arange(image_size), np.arange(image_size))

#Perform matrix operations to randomize spot localization
for _ in range(num_molecules):
    xo = np.random.randint(0, image_size)
    yo = np.random.randint(0, image_size)
    image += gauss2d(x, y, xo, yo)
