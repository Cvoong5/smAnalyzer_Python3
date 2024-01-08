import numpy as np
import pandas as pd
import tifffile
import nd2
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#== Reading in movies, return movie, summed_image
def imread(file_name):
    print("Reading data")
    file_ext = file_name.split('.')[1]
    if ".tif" in file_name:
        data = tifffile.imread(file_name)
    elif ".nd2" in file_name:
        data = nd2.imread(file_name)
    print("Summing data")
    summed_data = np.sum(data, axis = 0)
    uint16_data =((summed_data/np.max(summed_data))*(2**16-1)).astype(np.uint16)
    return data, uint16_data
#== Finding smCoordinates, return coordinates
def detect(data, threshold = 0):
    row, col = data.shape
    coordinates = []
    for y in range(1, row - 1):
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
                    for y2 in range(row - 1):
                        y2mid = data[y2, x]
                        y2high = data[y2 + 1, x]
                        if y2mid > y2high and y2mid > threshold:
                            coordinates.append((y2, x))
            elif xmid > xlow and xmid == xhigh and xmid > threshold:
                if ymid > ylow and ymid > yhigh and ymid > threshold:
                    for x2 in range(col - 1):
                        x2mid = data[y, x2]
                        x2high = data[y, x2 + 1]
                        if x2mid > x2high and x2mid > threshold:
                            coordinates.append((y, x2))
    return coordinates
#== Gaussian function
def gauss2d(array, intensity, backgorund, yo, xo, ysd, xsd):
    yarray, xarray = array
    return background + intensity * np.exp(-(0.5) * (xarray - xo)**2/xsd**2 + (yarray - yo)**2/ysd**2)
#== Temporal Median FIltering, return background subtracted image and background
def TemporalMedianFilter(image, radius = 3):
    print("Performing temporal median filtering")
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
    background = np.zeros_like(image)
    for y in range(radius, row - radius - 1):
        print(f"row {y}")
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
                    break
                m0, lb0, ub0 = medhist_bnd(Data, Omid)
                m1, lb1, ub1 = medhist_repchk(Data, dk, m0, lb0, ub0, Omid)
                background[y - radius : y + radius + 1, x - radius : x + radius + 1] = m1

    bg_sub = np.zeros_like(image)
    print("Performing background subtract")
    for yb in range(row):
        for xb in range(col):
            if image[yb, xb] > background[yb, xb]:
                bg_sub[yb, xb] = image[yb, xb] - background[yb, xb]
            else:
                bg_sub[yb, xb] = 0
    return bg_sub, background
#== Extract gauss2d parameters from single molecule, return gauss2d input and covariance
def gauss2d_Fit(coordinates, radius = 3):
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


