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
        array = np.zeros(array_size, dtype = np.uint16)
    elif noise == True:
        noise = np.random.randint(0, intensity - 1)
        array = np.random.poisson(lam = noise, size = array_size).astype(np.uint16)
    row, col = array.shape
    x, y = np.meshgrid(np.arange(col), np.arange(row))
    data_type = array.dtype
#Perform matrix operations to randomize spot localization
    coordinates = []
    for idx in range(num_molecules):
        if vary_sigma == True:
            sigma = np.random.randint(0, 3)
            y_sd, x_sd = (sigma, sigma)
        if vary_sigma == False:
            y_sd, x_sd = (1, 1)
        yo, xo = (np.random.randint(0, row), np.random.randint(0, col))
        if vary_intensity == True:
            variable_intensity = np.random.randint(0, intensity)
            gauss = gauss2d([y, x], variable_intensity, background, yo, xo, y_sd, x_sd).astype(data_type)
        elif vary_intensity == False:
            gauss = gauss2d([y, x], intensity, background, yo, xo, y_sd, x_sd).astype(data_type)
        array += gauss
        coordinates.append((yo, xo))
    return array, coordinates
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
#== Temporal Median Filter for background subtraction
def TemporalMedianFilter(image, radius = 3):
#== Core functions
    def medhist_bnd(D, Omid):
        Dmin = np.min(D)
        Dmax = np.max(D)
        try:
            count, intensity_bin = np.histogram(D, bins = np.arange(Dmin, Dmax))
            csum = 0
            #== This section of the code does not seem right
            for m in range(len(count)):
                if count[m] > 0:
                    csum += count[m]
                    if csum >= Omid:
                        mi = intensity_bin[m]
                        lb = csum - count[m] + 1
                        ub = csum
                        break
            #==
                else:
                    pass
        except TypeError:
            pass
        return mi, lb, ub
    def repchk(D, dk, m, lb, ub):

        d0 = D[0]

        if d0 < m and dk < m:
            lb1 = lb
            ub1 = ub
            
        elif d0 < m and dk == m:
            lb1 = lb - 1
            ub1 = ub

        elif d0 < m and dk > m:
            lb1 = lb - 1
            ub1 = ub - 1

        elif d0 == m and dk < m:
            lb1 = lb + 1
            ub1 = ub

        elif d0 == m and dk == m:
            lb1 = lb 
            ub1 = ub

        elif d0 == m and dk > m:
            lb1 = lb 
            ub1 = ub - 1

        elif d0 > m and dk < m:
            lb1 = lb + 1
            ub1 = ub + 1

        elif d0 > m and dk == m:
            lb1 = lb 
            ub1 = ub + 1

        elif d0 > m and dk > m:
            lb1 = lb 
            ub1 = ub 

        if lb1 <= Omid and Omid <= ub1:
            tf = 1
        else:
            tf = 0

        return tf, lb1, ub1
    def medhist_repchk(D, dk, m, lb, ub, Omid):
        D.append(dk)
        D1 = D
        D1.pop(0)
         ##Needs work
        tf, lb1, ub1 = repchk(D, dk, m, lb, ub)
        if tf == 1:
            m1 = m
        elif tf == 0:
            m1, lb1, ub1 = medhist_bnd(D1, Omid)
        return m1, lb1, ub1
#== Start of filter
    row, col = image.shape
    filtered_image = np.zeros_like(image)
    test = filtered_image.copy()
    for y in range(radius, row - radius - 1):
        for x in range(radius, col - radius - 1):
            image_slice = image[y - radius : y + radius + 1, x - radius : x + radius + 1]
            if image_slice.shape != (2*radius + 1, 2*radius + 1):
                print(image_slice.shape)
                continue
            else:
                flatten_image = np.ravel(image_slice).tolist()
                dk = flatten_image[-1]
                Data = flatten_image[:-1]
                N_count = len(flatten_image)
                N_count2 = int(np.sqrt(N_count))
                if N_count2%2 != 0:
                    Omid = (N_count - 1)/2
                elif N_count2%2 == 0:
                    print(f"N is an even number {N_count2}, please select an odd number")
                m0, lb0, ub0 = medhist_bnd(Data, Omid)
                m1, lb1, ub1 = medhist_repchk(Data, dk, m0, lb0, ub0, Omid)
                filtered_image[y - radius : y + radius, x - radius : x + radius] = m1

    bg_sub = np.zeros_like(image)
    for yb in range(row):
        for xb in range(col):
            if data[yb, xb] > filtered_image[yb, xb]:
                bg_sub[yb, xb] = image[yb, xb] - filtered_image[yb, xb]
            else:
                bg_sub[yb, xb] = 0
    return bg_sub, filtered_image
#== Fitting single molecules with gauss2d to extract information
def gauss2d_smFit(coordinates, radius = 3):
    slice_shape = (2*radius + 1, 2*radius + 1)
    fit_set = []
    for coord in coordinates:
        y, x = coord
        Z = data[y - radius: y + radius + 1, x - radius: x + radius + 1]
        if Z.shape == slice_shape:
            X, Y = np.meshgrid(np.arange(y - radius, y + radius + 1), np.arange(x - radius, x + radius + 1))
            try:
                estimate = [0, 0, y, x, 2, 2] #Intensity, Background, Y center, X center, Y std, X std
                fit_set.append(curve_fit(f = gauss2d, xdata = (np.ravel(X), np.ravel(Y)), ydata = np.ravel(Z), p0 = estimate))
            except RuntimeError:
                print(f"Could not fit {coord}")
        else:
            print(f"{coord} too close edge of image")
            continue
        
    return fit_set #[[Parameters], [Covariances]]
#== Drawing circles based on gaussian fit
def gauss2dFit_draw_circles(data, fit):
    fig, ax = plt.subplots()
    ax.imshow(data)
    for fit_info in fit_set:
        parameters, covariance = fit_info
        intensity, background, yo, xo, ysd, xsd = parameters
        circ = plt.Circle((xo, yo), radius = 3, fill = False, color = 'red')
        ax.add_patch(circ)
    plt.show()
#== Generate dynamics at fixed coordinates
#== Testing functions
np.random.seed(1234)
data, coordinates_gt = generate_random_molecules()
bg_sub, bg = TemporalMedianFilter(data, radius = 3)
coordinates_est1 = detect_molecules(bg_sub, threshold = 25)
fit_set = gauss2d_smFit(coordinates_est1)
gauss2dFit_draw_circles(bg_sub, fit_set)

    


