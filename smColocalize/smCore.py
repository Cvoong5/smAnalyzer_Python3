import numpy as np
import pandas as pd
import tifffile
import nd2
import matplotlib.pyplot as plt

def read_movie(file):
    print("Reading data")
    file_ext = file.split('.')[1]
    if file_ext == "tif": 
        movie = tifffile.imread(file)
    elif file_ext == "nd2":
        movie = nd2.imread(file)
    return movie
def sum_movie(movie):
    print("Summing data")
    return np.sum(movie, axis = 0)
def find_spots(image, threshold, dist_limit = 3, background_subtract = True, spot_size = 3):
    #Needs improvement#
        #Spot finding parameter based on local maxima, but also consider circularity.
    print("Finding spots")
    row, col = image.shape
    initial_coordinates = []
    final_coordinates = []  
    multiplier = 1000
    mthreshold = multiplier*threshold
    image_bgsub = np.zeros_like(image)
    if background_subtract == True:
        for y in range(spot_size, row - spot_size - 1):
            for x in range(spot_size, col - spot_size - 1):
                image_temp = image[y - spot_size:y + spot_size + 1, x - spot_size:x + spot_size + 1]
                image_bgsub[y - spot_size:y + spot_size + 1, x - spot_size: x + spot_size + 1] = image_temp - np.min(image_temp)
        image = image_bgsub
    for y in range(1, row - 1):
        for x in range(1, col - 1):
            row0 = image[y - 1, x]
            col0 = image[y, x - 1]
            pt = image[y, x ]
            row1 = image[y + 1, x]
            col1 = image[y, x + 1]
            if pt > mthreshold and pt > row0 and pt > row1 and pt > col0 and pt > col1:
                initial_coordinates.append((y, x))
    for set1 in initial_coordinates:
        y1, x1 = set1
        for set2 in initial_coordinates:
            y2, x2 = set2
            distance = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            save = False
            if distance != 0:
                if distance >= dist_limit:
                    save = True
                elif distance < dist_limit:
                    break
        if save == True:
            final_coordinates.append(set1)
    print(f"Number of spots: {len(final_coordinates)}")
    return final_coordinates #saved as (y, x) or (row, column)

def extract_intensity(image, movie, coordinates, window_size = 10, smooth_window = 3, spot_size = 1, sigma = 1):
    
    print("Extracting intensity traces")
    intensity_data = []
    counter = 0
    saved_coordinates = []
    n = len(coordinates)
    def box_car_smoothing(intensity, smooth_window):
        boxcar = []
        for i in range(len(intensity) - smooth_window):
            boxcar.append(int(np.mean(intensity[i: i + smooth_window])))
        boxcar = np.pad(boxcar, (0, smooth_window), mode = 'edge')
        return boxcar
#Currently, the fit_intensity parameter is struggling the most at the moment. I need to figure out a way to improve the fitting parameter ... #
    def fit_intensity(intensity, window, sigma):
        print("fitting intensity traces")
        mean_data = int(np.mean(intensity))
        std_data = int(np.std(intensity))

        mean_list = []
        std_list = []
        for i in range(0 , len(intensity) - window, window):
            mean_window = np.mean(intensity[i: i + window])
            std_window = np.std(intensity[i: i + window])
            if mean_window < 2*mean_data and std_window < 2*std_data:
                mean_list.append(mean_window)
                std_list.append(std_window)
        mean_threshold = int(np.mean(mean_list))
        std_threshold = int(np.mean(std_list))

        change_points = [0]
        for j in range(window, len(intensity) - window, window):
            range1 = intensity[j - window: j]
            range2 = intensity[j: j + window]
            mean1 = int(np.mean(range1))
            mean2 = int(np.mean(range2))
            mean_difference = mean2 - mean1

            if mean2 > mean1 and mean_difference > sigma*std_threshold and mean_difference > 0.2*mean2:
                change_points.append(j)
            elif mean1 > mean2 and abs(mean_difference) > sigma*std_threshold and abs(mean_difference) > 0.2*mean1:
                change_points.append(j)
        change_points.append(len(intensity))

        fit = np.zeros_like(intensity)
        for k in range(len(change_points) - 1):
            range1 = change_points[k]
            range2 = change_points[k + 1]
            fit[range1:range2] = int(np.mean(intensity[range1:range2]))
        update = True
        while update:
            update = False
            for l in range(len(fit) - 1):
                range1 = fit[l]
                range2 = fit[l+1]
                difference = range2 - range1
                if difference != 0:
                    if difference > 0:
                        if 0.5*range2 < range1:
                            fit[l] = fit[l+1]
                            update = True
                    elif difference < 0:
                        if 0.5*range1 < range2:
                            fit[l+1] = fit[l]
                            update = True
        return fit
    fig = plt.figure(figsize = (17, 9))
    fig.subplots_adjust(left = 0.05, right = 0.95, bottom = 0.05, top = 0.95, wspace = 0.05, hspace = 0.25)
    #fig2 = plt.figure(figsize = (3,3))
    #ax1 = fig2.add_subplot(111)
    #ax1.imshow(image, cmap = 'gray')
    #ax1.set_title("Image")
    ax2 = fig.add_subplot(221)
    ax3 = fig.add_subplot(223, projection = '3d')
    ax4 = fig.add_subplot(222)
    ax5 = fig.add_subplot(224)
    for coordinate in coordinates:
        counter += 1
        print(f"processing {counter} of {len(coordinates)}")
        data = []
        y, x = coordinate
        frame_number = 0
        spot_frame1 = image[y - spot_size: y + spot_size + 1, x - spot_size: x + spot_size + 1] 
        spot_frame2 = image[y - 2*spot_size: y + 2*spot_size + 1, x - 2*spot_size: x + 2*spot_size + 1]
        expanded_spot_frame1 = np.pad(spot_frame1, ((spot_size, spot_size)), mode = 'constant', constant_values = 0)
        bg = spot_frame2 - expanded_spot_frame1
        bg_int = np.mean(bg[bg > 0])
        spot_image = spot_frame2 - bg_int  
        if spot_image.shape == (4*spot_size + 1, 4*spot_size + 1):
            for frame in movie:
                frame_number += 1
                spot_movie = frame[y - spot_size: y + spot_size + 1, x - spot_size: x + spot_size + 1] 
                spot_bg = frame[y - 2*spot_size: y + 2*spot_size + 1, x - 2*spot_size: x + 2*spot_size + 1]
                spot_expand = np.pad(spot_movie, ((spot_size, spot_size), (spot_size, spot_size)), mode = 'constant', constant_values = 0)
                bg_diff = spot_bg - spot_expand
                bg_intensity = round(np.mean(bg_diff[bg_diff !=0]))
                spot_intensity = round(np.mean(spot_movie))
                passed_coordinate = coordinate
                data.append({'Frame': frame_number, 'Spot_intensity':spot_intensity, 'Background_intensity': bg_intensity, 'Background_subtract': spot_intensity - bg_intensity})
        data_df = pd.DataFrame(data)
        smooth = box_car_smoothing(data_df['Background_subtract'], smooth_window)
        fit = fit_intensity(data_df['Background_subtract'], window_size, sigma)
        fit_smooth = fit_intensity(smooth, smooth_window*window_size, sigma)
        data_df['Fit_intensity'] = fit
        data_df['Smooth_intensity'] = smooth
        data_df['Smooth_fit'] = fit_smooth
        #Plotting
        ystr, xstr = np.meshgrid(np.arange(4*spot_size + 1), np.arange(4*spot_size + 1))

        circles = plt.Circle((x,y), radius = 3, edgecolor = "red", fill = False, linewidth = 1)

        ax2.clear()
        ax3.clear()
        ax4.clear()
        ax5.clear()
        #ax1.add_artist(circles)
        #ax1.axis('off')
        ax2.imshow(spot_image, cmap = 'gray')
        ax2.axis('off')
        ax2.set_title(f"Spot {y}_{x}")
        ax3.plot_surface(xstr, ystr, spot_image)
        ax4.plot(data_df['Background_subtract'], color = 'black', label = "Data")
        ax4.plot(fit, color = 'red', label = "Fit")
        ax4.set_ylabel("Intensity")
        ax4.set_xlabel("Frame (Original)")
        ax5.plot(smooth, color = 'black', label = "Smooth data")
        ax5.plot(fit_smooth, color ='red',label = "Fit")
        ax5.set_ylabel("Intensity")
        ax5.set_xlabel(f"Frame (Boxcar smoothed: {smooth_window} frames)")
        plt.pause(0.5)
        plt.draw()
        save_spot = input("Save spot? (y/n)")
        if save_spot == "y":
            data_df.to_csv(f"spot_{y}_{x}.csv", index = False)
            plt.savefig(f"spot_{y}_{x}.png", dpi = 300, format = 'png', bbox_inches = 'tight')
        elif save_spot == "n":
            pass
    return

