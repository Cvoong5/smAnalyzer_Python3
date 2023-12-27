import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.optimize import curve_fit

#Core function to generate single molecules
def gauss2d(data_array, intensity, background, yo, xo, y_sd, x_sd):
    y_array, x_array = data_array
    return background + intensity * np.exp( -(1/2) * ( (x_array-xo)**2 / x_sd**2 + (y_array-yo)**2 / y_sd**2 ))
#smGenerators
#== Generates a single image of single molecules with the option to vary noise, intensity, number of molecules, and array size
def generate_random_molecules(num_molecules = 100, array_size = (512, 512), noise = True, vary_intensity = False, vary_sigma = False):
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
            y_sd, x_sd = (sigma, sigma)
        if vary_sigma == False:
            y_sd, x_sd = (sigma, sigma)

        yo, xo = (np.random.randint(0, row), np.random.randint(0, col))

        if vary_intensity == True:
            variable_intensity = np.random.randint(0, intensity)
            gauss = gauss2d([y, x], variable_intensity, background, yo, xo, y_sd, x_sd).astype(data_type)
        elif vary_intensity == False:
            gauss = gauss2d([y, x], intensity, background, yo, xo, y_sd, x_sd).astype(data_type)
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
def local_maxima(data, threshold = 0):
    row, col = data.shape
    coordinates = []
    for y in range(1, row -1):
        for x in range(1, col - 1):
            xlow = data[y, x - 1]
            xmid = data[y, x]
            xhigh = data[y, x + 1]
            ylow = data[y - 1, x]
            ymid = data[y, x]
            yhigh = data[y + 1, x]
            if xmid > xlow and xmid > xhigh and xmid > threshold:
                if ymid > ylow and ymid > yhigh and ymid > threshold:
                    coordinates.append((y,x))
                elif ymid > ylow and ymid == yhigh and ymid > threshold:
                    for y2 in range(1, row - 1):
                        y2mid = data[y2, x]
                        y2high = data[y2 + 1, x]
                        if y2mid > y2high and y2mid > threshold:
                            coordinates.append((y2, x))
            elif xmid > xlow and xmid == xhigh and xmid > threshold:
                for x2 in range(1, col - 1):
                    x2mid = data[y, x2]
                    x2high = data[y, x2 + 1]
                    if x2mid > x2high and x2mid > threshold:
                        coordinates.append((y, x2))
    return coordinates

def fit_gauss(data, coordinates):
    pass



np.random.seed(1234)
test_data = generate_random_molecules()

row, col = test_data.shape
radius = 3
background_sub = np.zeros_like(test_data)
for y in range(radius, row - radius):
    for x in range(radius, col - radius):
        section = test_data[y - radius : y + radius, x - radius : x + radius]
        background = np.mean(section)
        background_sub[y - radius : y + radius, x - radius : x + radius] = section - background


coordinates = local_maxima(background_sub, threshold = 15)

#== Gaussian fitting
intensity = 1
background = 1
sd = [1,1]

circles = []
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.imshow(test_data)
for idx in coordinates:
    y,x = idx
    intensity_val = test_data[y - radius: y + radius, x - radius: x + radius]
    if intensity_val.shape == (radius*2, radius*2):
        y_len,x_len = intensity_val.shape
        x_grid,y_grid = np.meshgrid(np.arange(x_len), np.arange(y_len))
        guess = [intensity, background, y, x, 1, 1]
        param, cov = curve_fit(f = gauss2d, xdata = np.ravel((512,512)), ydata = np.ravel(intensity_val), p0 = guess)
        i, b, yo, xo, y_sd, x_sd = param
        print(yo, xo)
        pass
    circle = plt.Circle((x,y), radius = 3, fill = False, edgecolor = "red")
    ax.add_patch(circle)
plt.show()

#==To develop
#===Fitting detected single molecules with a 2D Gaussian as a way to:
#==== Find the true center
#==== Smooth single molecule if there isn't a defined maxima within the standard deviation (2 peaks nearby and are not categorized as two separate peaks)
#==== Optional accept or reject spot based on characteristics of single molecule
