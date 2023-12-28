import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.optimize import curve_fit

#Core function to generate single molecules
def gauss2d(data_array, intensity, background, center, st_deviation):
    y_array, x_array = data_array
    yo, xo = center
    x_sd, y_sd = st_deviation
    return background + intensity * np.exp( -(1/2) * ( (x_array-xo)**2 / x_sd**2 + (y_array-yo)**2 / y_sd**2 ))
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
            std_dev = (1, 1)
        center = (np.random.randint(0, row), np.random.randint(0, col))
        if vary_intensity == True:
            variable_intensity = np.random.randint(0, intensity)
            gauss = gauss2d([y, x], variable_intensity, background, center, std_dev).astype(data_type)
        elif vary_intensity == False:
            gauss = gauss2d([y, x], intensity, background, center, std_dev).astype(data_type)
        array += gauss
    return array
#== Generates the desired number of coordinates on a desired array
def generate_random_coordinates(num_coordinates = 100, array_size = (512, 512)):
    coordinates = np.zeros((num_coordinates, 2))
    for idx in range(num_coordinates):
        coordinates[idx] = (np.random.randint(512), np.random.randint(512))
    return coordinates
#== Detecting the local maxima of single molecules
def detect_molecules(data, threshold = 0, edge = 1):
    row, col = data.shape
    coordinates = []
    for y in range(edge, row - edge):
        for x in range(edge, col - edge):
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
                    for y2 in range(row - edge):
                        y2mid = data[y2, x]
                        y2high = data[y2 + edge, x]
                        if y2mid > y2high and y2mid > threshold:
                            coordinates.append((y2, x))
            elif xmid > xlow and xmid == xhigh and xmid > threshold:
                if ymid > ylow and ymid > yhigh and ymid > threshold:
                    for x2 in range(col - edge):
                        x2mid = data[y, x2]
                        x2high = data[y, x2 + edge]
                        if x2mid > x2high and x2mid > threshold:
                            coordinates.append((y, x2))
    return coordinates

def generate_dynamics():
    yx_coordinates = generate_random_coordinates(num_coordinates = 25)
    x, y = np.meshgrid(np.arange(512), np.arange(512))
    frames = 1000
    movie = np.zeros((frames, 512, 512)).astype(np.uint16)
    for frame in range(frames):
        progress = round(100*(frame/frames), 1)
        if frame % 10 == 0:
            print(f"{progress} %")
        image = np.zeros((512,512)).astype(np.uint16)
        array_shape = image.shape

        for idx in range(len(yx_coordinates)):
            binding = np.random.randint(0, 3) 
            if binding == 0:
               pass
            else:
               gauss = gauss2d((y, x), 1000, 0, yx_coordinates[idx], (1, 1)).astype(np.uint16)
               image += gauss
        mean_bg = 500
        background = np.random.randint(low = np.max(image)*1  , high = np.max(image)*2 , size = (512, 512)).astype(np.uint16)
        background += image
        movie[frame] = background

    tifffile.imwrite("test_movie.tif", movie, dtype = np.uint16)




