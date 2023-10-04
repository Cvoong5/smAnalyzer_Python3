import tifffile
import nd2
import os
import glob

location = input(f"Please enter the directory where your nd2 files are located\n")
os.chdir(location)

file_names = glob.glob("*.nd2")
for i in file_names:
    file_name = i.split('.')[0]
    tiff_name = f"{file_name}.tif"
    print(f"Creating {file_name}.tif from {i}")
    nd2file = nd2.imread(i)
    tifffile.imwrite(tiff_name, nd2file, imagej = True)
print("tiff file generation complete")
