import numpy as np
import pandas as pd
import tifffile
import nd2
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import tabulate

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
def gauss2d(array, intensity, background, yo, xo, ysd, xsd):
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
    counter = 0
    for y in range(radius, row - radius - 1):
        counter += 1
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
def gauss2d_Fit(data, coordinates, radius = 3):
    slice_shape = (2*radius + 1, 2*radius + 1)
    fit_set = []
    fit_coordinate = []
    for coord in coordinates:
        y, x = coord
        Z = data[y - radius: y + radius + 1, x - radius: x + radius + 1]
        if Z.shape == slice_shape:
            X, Y = np.meshgrid(np.arange(y - radius, y + radius + 1), np.arange(x - radius, x + radius + 1))
            try:
                estimate = [0, 0, y, x, 0, 0] #Intensity, Background, Y center, X center, Y std, X std
                fit = curve_fit(f = gauss2d, xdata = (np.ravel(X), np.ravel(Y)), ydata = np.ravel(Z), p0 = estimate)
                fit_set.append(fit)
                fit_coordinate.append((fit[0][2], fit[0][3]))
            except RuntimeError:
                fit_set.append([0,0,0,0,0,0], [0,0,0,0,0,0])
                fit_coordinate.append((0, 0))
                print(f"Could not fit {coord}")
        else:
            print(f"{coord} too close edge of image")
            continue
    return fit_set, fit_coordinate #[[Parameters], [Covariances]]
    return
def transform(image, angle, yoffset, xoffset, flip = None):
     #flip = h for horizontal or v for vertical 
        shifted_image = np.zeros_like(image)
        row, col = image.shape
        ycenter = row//2
        xcenter = col//2
        theta = np.radians(angle)
        print(f"Angle = {angle}\nx offset = {xoffset}\ny offset = {yoffset}\nflip = {flip}")
        if flip != None:
            if flip == "h" or "horizontal":
                image = np.flip(image, 1)
            elif flip == "v" or "vertical":
                image = np.flip(image, 0)
            else:
                print("invalid flip option")
                pass
        #Mathematical explanation for image rotation
            #Translation along the origin: x - xcenter, y - ycenter
            #Rotation transformation: x coordinates = (x translation)*cos(theta) - (y translation)*sin(theta), y coordinates = (x translation)*sin(theta) + (y translation)*cos(theta)
            #Reorient the coordinates: Rotation transformation on each axis + center coordinates
        for y in range(row):
                for x in range(col):
                                                #Translation, rotation, reorient, offset
                        x_shift = int(((x-xcenter)*np.cos(theta) - (y-ycenter)*np.sin(theta) + xcenter) + xoffset)
                        y_shift = int(((x-xcenter)*np.sin(theta) + (y-ycenter)*np.cos(theta) + ycenter) + yoffset)
                        if y_shift >= 0 and y_shift < row and x_shift >= 0 and x_shift < col:
                                shifted_image[y, x] = image[y_shift, x_shift]
        return shifted_image
#== Extract information from movie at respective coordinates
    #== Add a visual way to look at intensity traces, summed spot, and spots at certain frames.
def lbp_filter(time_series):
    frame, intensity, background = time_series.columns
    #== Time series is a pandas data frame
    #=== Col#1 = frame, col#2 = intensity, col#3 = background
    
    #== Time series from extract_intensity in the form of a data frame
    lbp_list = []
    for index in range(1, len(time_series)):
        spot_frame = time_series.loc[index, frame]
        spot_intensity = time_series.loc[index, intensity] - time_series.loc[index - 1, intensity]
        spot_background = time_series.loc[index, background] - time_series.loc[index - 1, background]
        dictionary = {frame: spot_frame, intensity: spot_intensity, background: spot_background}
        lbp_list.append(dictionary)
    df = pd.DataFrame(lbp_list)
    return df 

def extract(data, coordinates, radius = 3, sigma = 1):
    #== Data has to be the movie in which we are aiming to extract the intensity from. The coordinates can come from either the local maxima detection or the gaussian non-linear least squares fitting algorithm.
    print("Extracting intensity values")
    frame, row, col = data.shape
    pad = int(radius - sigma)
    #If full window size is 11 and spot window size occupies the center 3, what is the range?
        #N-1/2 +/- 1
         # [0, 1, 2, 3, 4, 5, 6, 7 ,8 ,9 , 10]
    frames = np.arange(frame)
    summed_image = np.sum(data, axis = 0)
    X, Y = np.meshgrid(np.arange(2*radius + 1), np.arange(2*radius + 1))
    df_set = []

    fig = plt.figure()
    ax_spot = fig.add_subplot(321)
    ax_surface = fig.add_subplot(322, projection = '3d')
    ax_traces = fig.add_subplot(312)
    ax_filtered = fig.add_subplot(313)
    if np.min(data) == 0:
        data += 1
    for coord in coordinates:
        yo, xo =  coord 
        y = int(yo)
        x = int(xo)
        intensity_vals = []
        try:
            summed_spot = summed_image[y - radius : y + radius + 1, x - radius : x + radius + 1]
            for z in range(frame):
                spot_full = data[z, y - radius : y + radius + 1, x - radius : x + radius + 1]
                spot_window = np.pad(array = data[z, y - sigma: y + sigma + 1, x - sigma: x + sigma + 1], pad_width = pad, mode = 'constant', constant_values = 0)
                background = spot_full - spot_window
                spot_intensity = np.mean(spot_window[spot_window != 0])
                background_intensity = np.mean(background[background != 0])
                intensity_vals.append({"frame": z + 1, "spot_intensity": spot_intensity, "background": background_intensity})

                #Average to fit within 10 frames
                #3D Gaussian
                #Summed_image
        except (TypeError, ValueError):
            if TypeError:
                print("Type error")
                continue
            elif ValueError:
                print("Shape error")
                continue
            continue

        data_frame = pd.DataFrame(intensity_vals)
        df_set.append(data_frame)
        filtered_data = lbp_filter(data_frame)
#        data_frame.to_csv(f"spot_{y}_{x}.csv", index = False)

        ax_spot.clear()
        ax_surface.clear()
        ax_traces.clear()
        ax_filtered.clear()
        fig.suptitle(f"spot_{y}_{x}")
        ax_spot.imshow(summed_spot, cmap = "gray")
        ax_spot.axis('off')
        ax_surface.plot_surface(X, Y, summed_spot, cmap = "hot")
        ax_spot.axis('off')
        ax_traces.plot(data_frame["frame"], data_frame["spot_intensity"] - data_frame["background"])
        ax_traces.set_xlabel("Frames")
        ax_traces.set_ylabel("Intensity (bg subtracted)")
        ax_filtered.plot(filtered_data["frame"], filtered_data["spot_intensity"] - filtered_data["background"])
        ax_filtered.set_xlabel("Frames")
        ax_filtered.set_ylabel("Intensity (bg subtracted)")
#        #==Uncomment the following line of code for testing purposes
        plt.draw()
        plt.pause(1)
        #==Uncomment the following line of code for final implementation
        #plt.savefig(f"spot_{y}_{x}.jpg")

    return df_set 



