import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.optimize import curve_fit

#Core function to generate single molecules
def gauss2d(data_array, intensity, background, center, st_deviation):
    y_array, x_array = data_array
    yo, xo = center
    x_sd, y_sd = st_deviation
    return background + intensity * np.exp( -(1/2) * ( (x_array-xo)**2 / x_sd + (y_array-yo)**2 / y_sd ))
#smGenerators
#== Generates a single image of single molecules with the option to vary noise, intensity, number of molecules, and array size
def generate_random_molecules(num_molecules = 100, array_size = (512, 512), noise = True, vary_intensity = True, vary_sigma = True):
# Create an empty 512x512 image array 
    background = 0
    intensity = np.random.randint(1, 100)
    sigma = np.random.randint(0, 3)
    if noise == False:
        array = np.zeros(array_size)
    elif noise == True:
        noise = np.random.randint(0, intensity - 1)
        array = np.random.poisson(lam = noise, size = array_size)
    row, col = array.shape
    x, y = np.meshgrid(np.arange(col), np.arange(row))
    data_type = array.dtype
#Perform matrix operations to randomize spot localization
    for idx in range(num_molecules):
        if vary_sigma == True:
            sigma = np.random.randint(0, 3)
            std_dev = (sigma, sigma)
        if vary_sigma == False:
            std_dev = (sigma, sigma)
        center = (np.random.randint(0, row), np.random.randint(0, col))
        if vary_intensity == True:
            variable_intensity = np.random.randint(0, intensity)
            gauss = gauss2d([y, x], variable_intensity, background, center, std_dev).astype(data_type)
        elif vary_intensity == False:
            gauss = gauss2d([y, x], intensity, background, center, std_dev).astype(data_type)
        true_yx[idx] = center
        array += gauss
    return array
#== Generates the desired number of coordinates on a desired array
def generate_random_coordinates(num_coordinates = 100, array_size = (512, 512)):
    coordinates = np.zeros((num_coordinates, 2))
    for idx in range(num_coordinates):
        coordinates[idx] = (np.random.randint(512), np.random.randint(512))
    return coordinates
#== Generates background noise via the poisson distribution method
def generate_background(array_size = (512, 512), limit = 1):
    return np.random.poisson(lam = limit, size = array_size)
#== Detecting the local maxima of single molecules
def detect_coordinates_local_maxima(data):
    row, col = data.shape
    coordinates = []
    for y in range(6, row -6):
        for x in range(6, col - 6):
            xlow = np.sum(data[y, x - 6 : x + 6])
            xmid = np.sum(data[y, x - 3 : x + 3])
            xhigh = np.sum(data[y, x + 3 : x + 6])
            ylow = np.sum(data[y - 6 : y + 6, x])
            ymid = np.sum(data[y - 3 : y + 3, x])
            yhigh = np.sum(data[y + 3 : y + 6, x])
            if xmid > xlow and xmid > xhigh:
                if ymid > ylow and ymid > yhigh:
                    coordinates.append((y,x))
                elif ymid > ylow and ymid == yhigh:
                    for y2 in range(3, row - 6):
                        y2mid = np.sum(data[y2 - 3 : y2 + 3, x])
                        y2high = np.sum(data[y2 + 3: y + 6, x])
                        if y2mid > y2high:
                            coordinates.append((y2, x))
            elif xmid > xlow and xmid == xhigh:
                for x2 in range(3, col - 6):
                    x2mid = np.sum(data[y, x2 - 3 : x2 + 3])
                    x2high = np.sum(data[y, x2 + 3 : x2 + 6])
                    if x2mid > x2high:
                        coordinates.append((y, x2))






#==Development
#=== Trying to develop a stack with randomize on/off intensities at defined coordinates in a noise-simulated environment
yx_coordinates = generate_random_coordinates(num_coordinates = 1000)
background = generate_background(limit = 1000).astype(np.int64)

row, col = background.shape
x, y = np.meshgrid(np.arange(col), np.arange(row))
for idx in range(len(yx_coordinates)):
    gauss = gauss2d((y, x), 500, 0, yx_coordinates[idx], (1, 1)).astype(np.int64)
    background += gauss

plt.imshow(background)
plt.show()

#==To develop
#===Fitting detected single molecules with a 2D Gaussian as a way to:
#==== Find the true center
#==== Smooth single molecule if there isn't a defined maxima within the standard deviation (2 peaks nearby and are not categorized as two separate peaks)
#==== Optional accept or reject spot based on characteristics of single molecule
