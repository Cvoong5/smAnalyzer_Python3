#smColocalize.py
import numpy as np
import pandas as pd
import smCore
import smSupport
import math
import matplotlib.pyplot as plt

def quantify_colocalization(coordinates1, coordinates2, delta = 3):
    count = 0
    for coord1 in coordinates1:
        y1, x1 = coord1
        for coord2 in coordinates2:
            y2, x2 = coord2
            dist = round(np.sqrt((y2 - y1)**2 + (x2 - x1)**2))
            if dist <= delta:
                count += 1
    return count
def assess_colocalization(coordinates1, coordinates2, save_coordinates = "y"):
    colocalized_coordinates = []
    for c1 in coordinates1:
        y1, x1 = c1
        for c2 in coordinates2:
            y2, x2 = c2
            distance = round(np.sqrt((y2 - y1)**2 + (x2 - x1)**2))
            if distance <= 5:
                saved_coordinates = {"distance": distance, "coordinates1": c1, "coordinates2": c2}
                colocalized_coordinates.append(saved_coordinates)
    cc_df = pd.DataFrame(colocalized_coordinates).sort_values(by = 'distance').reset_index(drop=True)
    for delta in range(6):
        l_delta = len(cc_df[cc_df['distance'] <= delta])
        print(l_delta)
    if save_coordinates == "y":
        cc_df.to_csv("Saved_coordinates.csv", index = False)
    return cc_df
def get_offset(file1, file2, threshold1, threshold2):
    #Need to incorporate a way to expand or shrink an image#
    #Reading movies
    movie1, image1 = smCore.read_data(file1)
    movie2, image2 = smCore.read_data(file2)
    
    #Viewing images
        #Initial spot finding
    coordinates1 = smCore.find_spots(image1, threshold1)
    coordinates2 = smCore.find_spots(image2, threshold2)



    plt.figure(figsize = (5,5))
    y1 = [y[0] for y in coordinates1]
    x1 = [x[1] for x in coordinates1]
    max_rotate = 0
    max_rotate2 = 0
    max_x = 0
    max_x2 = 0
    max_y = 0
    max_y2 = 0
    opt_rotate = 0
    opt_rotate2 = 0
    opt_x = 0
    opt_x2 = 0
    opt_y = 0
    opt_y2 = 0
    for m in range(-20 , 20):
        #Rotate
        rcoord2 = smSupport.shift_coordinates(image2.shape, coordinates2, m, 0, 0, mode = 'list')
        count = quantify_colocalization(coordinates1, rcoord2, delta = 3)
        if count > max_rotate:
            max_rotate = count
            opt_rotate = m
        #Seperate the coordinates
        y2 = [y[0] for y in rcoord2]
        x2 = [x[1] for x in rcoord2]
        #View the coordinates
        plt.clf()
        plt.axis('off')
        plt.xlim(0, 511)
        plt.ylim(0, 511)
        plt.scatter(x1, y1, color = 'black', s = 0.5)
        plt.scatter(x2, y2, color = 'red', s = 0.5)
        plt.draw()
        plt.pause(0.06)
    print(opt_rotate)
    for k in range(-20 , 20):
        #Rotate
        rcoord2 = smSupport.shift_coordinates(image2.shape, coordinates2, opt_rotate, k, 0, mode = 'list')
        count = quantify_colocalization(coordinates1, rcoord2, delta = 3)
        if count > max_x:
            max_x = count
            opt_x = k
        #Seperate the coordinates
        y2 = [y[0] for y in rcoord2]
        x2 = [x[1] for x in rcoord2]
        #View the coordinates
        plt.clf()
        plt.axis('off')
        plt.xlim(0, 511)
        plt.ylim(0, 511)
        plt.scatter(x1, y1, color = 'black', s = 0.5)
        plt.scatter(x2, y2, color = 'red', s = 0.5)
        plt.draw()
        plt.pause(0.06)
    print(opt_x)
    for l in range(-20 , 20):
        #Rotate
        rcoord2 = smSupport.shift_coordinates(image2.shape, coordinates2, opt_rotate, opt_x, l, mode = 'list')
        count = quantify_colocalization(coordinates1, rcoord2, delta = 3)
        if count > max_y:
            max_y = count
            opt_y = l
        #Seperate the coordinates
        y2 = [y[0] for y in rcoord2]
        x2 = [x[1] for x in rcoord2]
        #View the coordinates
        plt.clf()
        plt.axis('off')
        plt.xlim(0, 511)
        plt.ylim(0, 511)
        plt.scatter(x1, y1, color = 'black', s = 0.5)
        plt.scatter(x2, y2, color = 'red', s = 0.5)
        plt.draw()
        plt.pause(0.06)
    print(opt_y) 
    for m in range(-20 , 20):
        rcoord2 = smSupport.shift_coordinates(image2.shape, coordinates2, m, opt_x, opt_y, mode = 'list')
        count = quantify_colocalization(coordinates1, rcoord2, delta = 3)
        if count > max_rotate2:
            max_rotate2 = count
            opt_rotate2 = m
        #Seperate the coordinates
        y2 = [y[0] for y in rcoord2]
        x2 = [x[1] for x in rcoord2]
        #View the coordinates
        plt.clf()
        plt.axis('off')
        plt.xlim(0, 511)
        plt.ylim(0, 511)
        plt.scatter(x1, y1, color = 'black', s = 0.5)
        plt.scatter(x2, y2, color = 'red', s = 0.5)
        plt.draw()
        plt.pause(0.06)
    print(opt_rotate2)
    for m in range(-20 , 20):
        rcoord2 = smSupport.shift_coordinates(image2.shape, coordinates2, opt_rotate2, m, opt_y, mode = 'list')
        count = quantify_colocalization(coordinates1, rcoord2, delta = 0)
        if count > max_x2:
            max_x2 = count
            opt_x2 = m
        #Seperate the coordinates
        y2 = [y[0] for y in rcoord2]
        x2 = [x[1] for x in rcoord2]
        #View the coordinates
        plt.clf()
        plt.axis('off')
        plt.xlim(0, 511)
        plt.ylim(0, 511)
        plt.scatter(x1, y1, color = 'black', s = 0.5)
        plt.scatter(x2, y2, color = 'red', s = 0.5)
        plt.draw()
        plt.pause(0.06)
    print(opt_x2)
    for m in range(-20 , 20):
        rcoord2 = smSupport.shift_coordinates(image2.shape, coordinates2, opt_rotate2, opt_x2, m, mode = 'list')
        count = quantify_colocalization(coordinates1, rcoord2, delta = 0)
        if count > max_y2:
            max_y2 = count
            opt_y2 = m
        #Seperate the coordinates
        y2 = [y[0] for y in rcoord2]
        x2 = [x[1] for x in rcoord2]
        #View the coordinates
        plt.clf()
        plt.axis('off')
        plt.xlim(0, 511)
        plt.ylim(0, 511)
        plt.scatter(x1, y1, color = 'black', s = 0.5)
        plt.scatter(x2, y2, color = 'red', s = 0.5)
        plt.draw()
        plt.pause(0.06)
    print(opt_y2)
    print(f"Optimal parameters: Rotation {opt_rotate2}, x offset {opt_x2}, y offset {opt_y2}")
        
    rcoord2 = smSupport.shift_coordinates(image2.shape, coordinates2, opt_rotate2, opt_x2, opt_y2, mode = 'list')
    y2 = [y[0] for y in rcoord2]
    x2 = [x[1] for x in rcoord2]
    
    plt.clf()
    plt.axis('off')
    plt.xlim(0, 511)
    plt.ylim(0, 511)
    plt.scatter(x1, y1, color = 'black', s = 0.5)
    plt.scatter(x2, y2, color = 'red', s = 0.5)
    plt.draw()
    plt.show()

    assess_colocalization(coordinates1, rcoord2, save_coordinates = "y")
    return
    movie1, image1 = smCore.read_data(file1)
    movie2, image2 = smCore.read_data(file2)
def overlay_images(image1, image2):
    #Layering images using alpha blending
   #extent = np.min(image1), np.max(image1), np.min(image2), np.max(image2)
    fig = plt.figure(frameon = False)
    im1 = plt.imshow(image1, cmap = plt.cm.viridis, interpolation = 'nearest', vmin = np.min(image1), vmax = np.max(image1))
    im2 = plt.imshow(image2, cmap = plt.cm.inferno, alpha = 0.5, interpolation = 'bilinear', vmin = np.min(image2), vmax = np.max(image2))
    plt.axis('off')
    plt.show()
    return
