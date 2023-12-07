import os
import numpy as np
import pandas as pd
import tifffile
import nd2
import glob
import matplotlib.pyplot as plt

def nd2_to_tif():
    directory = input("Enter path directory")
    os.chdir(directory)
    file_names = glob.glob("*.nd2")
    for file_name in file_names:
        fname = f"{file_name.split('.')[0]}.tif"
        print(f"Converting {file_name} to {fname}")
        movie = nd2.imread(file_name)
        tifffile.imwrite(fname, movie, imagej = True)
    print("ND2 to Tif conversion is complete") 
def draw_circles(image, coordinates, color, min_int, max_int):
    if not isinstance(image, list):
        image = [image]
        coordinates = [coordinates]
        color = [color]
    n = len(image)

    fig, axs = plt.subplots(1, n)
    if n == 1:
        axs = [axs]
    for i in range(n):
        img = image[i]
        coord = coordinates[i]
        col = color[i]
        axs[i].imshow(img, cmap = 'gray', vmin = min_int, vmax = max_int)
        for coordinate in coord:
            y, x = coordinate
            axs[i].add_artist(plt.Circle((x, y), radius = 3, edgecolor = col, fill = False, linewidth = 1))
    plt.show()
def display_image(image):
    if not isinstance(image, list):
        image = [image]
    n = len(image)  
    fig, axs = plt.subplots(1, n)
    if n == 1:
        axs = [axs]
    for i in range(n):
        img = image[i]
        axs[i].imshow(img, cmap = 'gray')
    plt.show()

def offset_image(image, angle = 0, y_offset = 0, x_offset = 0, flip = None):
        #flip = h for horizontal or v for vertical 
        shifted_image = np.zeros_like(image)
        row, col = image.shape
        ycenter = row//2
        xcenter = col//2
        theta = np.radians(angle)
        print(f"Angle = {angle}\nx offset = {x_offset}\ny offset = {y_offset}\nflip = {flip}")
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
                        x_shift = int(((x-xcenter)*np.cos(theta) - (y-ycenter)*np.sin(theta) + xcenter) + x_offset)
                        y_shift = int(((x-xcenter)*np.sin(theta) + (y-ycenter)*np.cos(theta) + ycenter) + y_offset)
                        if y_shift >= 0 and y_shift < row and x_shift >= 0 and x_shift < col:
                                shifted_image[y, x] = image[y_shift, x_shift]
        return shifted_image

def offset_coordinates(image_shape, coordinates, angle, y_offset, x_offset, mode = "list"):
    row, col = image_shape
    ycenter = row//2
    xcenter = col//2
    theta = np.radians(angle)
    shifted_coordinates = []
    if mode == "list":
        for coordinate in coordinates:
            y, x = coordinate
            x_shift = int(((x - xcenter)*np.cos(theta) - (y - ycenter)*np.sin(theta) + xcenter) + x_offset)
            y_shift = int(((x - xcenter)*np.sin(theta) + (y - ycenter)*np.cos(theta) + ycenter) + y_offset)
            if y_shift >= 0 and y_shift < row and x_shift >=0 and x_shift < col:
                shifted_coordinates.append((y_shift, x_shift))
        return shifted_coordinates 

    elif mode == "dataframe":
        for i in range(len(coordinates)):
            x = coordinates['x'][i]
            y = coordinates['y'][i]
            x_shift = int(((x - xcenter)*np.cos(theta) - (y - ycenter)*np.sin(theta) + xcenter) + x_offset)
            y_shift = int(((x - xcenter)*np.sin(theta) + (y - ycenter)*np.cos(theta) + ycenter) + y_offset)
            if y_shift >= 0 and y_shift < row and x_shift >=0 and x_shift < col:
                temp = {'x':x_shift, 'y':y_shift}
                shifted_coordinates.append(temp)
        shifted_coordinates_df = pd.DataFrame(shifted_coordinates)
        return shifted_coordinates_df
    
def coordinates_list_to_dataframe(coordinates):
    organized_coordinates = []
    for coordinate in coordinates:
        y, x = coordinate
        coordinates_separated = {'x': x, 'y':y}
        organized_coordinates.append(coordinates_separated)
    coordinates_df = pd.DataFrame(organized_coordinates)
    return coordinates_df



