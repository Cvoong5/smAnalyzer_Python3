from smSimulation import (gauss2d,
                          generate_random_molecules,
                          generate_random_coordinates,
                          detect_molecules)
import numpy as np
import matplotlib.pyplot as plt
import tifffile

#==Development
#=== Trying to develop a stack with randomize on/off intensities at defined coordinates in a noise-simulated environment
#yx_coordinates = generate_random_coordinates(num_coordinates = 25)
#x, y = np.meshgrid(np.arange(512), np.arange(512))
#frames = 1000
#movie = np.zeros((frames, 512, 512)).astype(np.uint16)
# in range(frames):
#    progress = round(100*(frame/frames), 1)
#    if frame % 10 == 0:
#        print(f"{progress} %")
#    image = np.zeros((512,512)).astype(np.uint16)
#    array_shape = image.shape
#
#    for idx in range(len(yx_coordinates)):
#        binding = np.random.randint(0, 3) 
#        if binding == 0:
#           pass
#        else:
#           gauss = gauss2d((y, x), 1000, 0, yx_coordinates[idx], (1, 1)).astype(np.uint16)
#           image += gauss
#    mean_bg = 500
#    background = np.random.randint(low = np.max(image)*1  , high = np.max(image)*2 , size = (512, 512)).astype(np.uint16)
#    background += image
#    movie[frame] = background
#
#
#tifffile.imwrite("test_movie.tif", movie, dtype = np.uint16)

for group in range(0, 1100, 200):
    image = generate_random_molecules(num_molecules = group, noise = False, vary_intensity = False, vary_sigma = False)
    plt.suptitle(f"{group}")
    plt.imshow(image)
    plt.draw()
    plt.pause(0.6)
plt.show()

#==To develop
#===Fitting detected single molecules with a 2D Gaussian as a way to:
#==== Find the true center
#==== Smooth single molecule if there isn't a defined maxima within the standard deviation (2 peaks nearby and are not categorized as two separate peaks)
#==== Optional accept or reject spot based on characteristics of single molecule

